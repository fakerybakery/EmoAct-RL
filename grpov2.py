import sys
import os
import numpy as np
import tempfile
import soundfile as sf
from majestrino_tagger import MajestrinoTagger

# =============================================================================
# ENVIRONMENT
# =============================================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import re
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from snac import SNAC
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    pipeline,
)
from trl import GRPOConfig, GRPOTrainer
from jiwer import wer as compute_wer

# =============================================================================
# CONFIG & PATHS
# =============================================================================
BASE_MODEL_NAME = "ChristophSchuhmann/Vocalino_0.11_alpha"
LOCAL_MODEL_PATH = "./vocalino_ct"
DATASET_NAME = "mrfakename/emoact_prompts_with_language"
SAMPLE_RATE = 24000

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"
SNAC_DEVICE = "cpu"  # CPU is fine, 300GB RAM

# =============================================================================
# VOCALINO TOKENS - THESE ARE THE CORRECT IDS THE MODEL WAS TRAINED WITH
# =============================================================================
TOKEN_OFFSET_BASE = 128266
SNAC_VOCAB_SIZE = 4096
LAYER_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]
AUDIO_TOKEN_END = 7 * SNAC_VOCAB_SIZE  # 28672

# Special token IDs - model was trained with these exact IDs
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_TEXT = 128009

# =============================================================================
# LOGGING
# =============================================================================
def log_debug(msg):
    if LOCAL_RANK == 0:
        print(f"[DEBUG] {msg}", flush=True)

# =============================================================================
# MODEL PREPARATION - Just save without modifying chat template
# =============================================================================
def prepare_local_model():
    marker_file = os.path.join(LOCAL_MODEL_PATH, ".ready_v3")
    if os.path.exists(marker_file):
        return
    if LOCAL_RANK == 0:
        log_debug("Preparing local model (preserving original tokenizer)...")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model_tmp = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16)
        tokenizer_tmp = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        # DON'T modify the tokenizer - keep original vocab and mappings
        model_tmp.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer_tmp.save_pretrained(LOCAL_MODEL_PATH)
        with open(marker_file, "w") as f:
            f.write("ready")
        log_debug(f"Saved model to {LOCAL_MODEL_PATH}")
    else:
        while not os.path.exists(marker_file):
            time.sleep(1)

prepare_local_model()

# =============================================================================
# TOKENIZER DIAGNOSTICS
# =============================================================================
def diagnose_tokenizer(tokenizer):
    """Check if the tokenizer can properly handle the special tokens."""
    log_debug("=" * 60)
    log_debug("TOKENIZER DIAGNOSTICS")
    log_debug("=" * 60)

    log_debug(f"Vocab size: {tokenizer.vocab_size}")
    log_debug(f"Len(tokenizer): {len(tokenizer)}")

    # Check what the special token IDs decode to
    special_ids = {
        "START_OF_SPEECH": START_OF_SPEECH,
        "END_OF_SPEECH": END_OF_SPEECH,
        "START_OF_HUMAN": START_OF_HUMAN,
        "END_OF_HUMAN": END_OF_HUMAN,
        "START_OF_AI": START_OF_AI,
    }

    for name, token_id in special_ids.items():
        try:
            decoded = tokenizer.decode([token_id])
            re_encoded = tokenizer.encode(decoded, add_special_tokens=False)
            log_debug(f"{name} (ID {token_id}): decodes to '{decoded}' -> re-encodes to {re_encoded}")
        except Exception as e:
            log_debug(f"{name} (ID {token_id}): ERROR - {e}")

    # Check audio token range
    sample_audio_ids = [128266, 128270, 130000, 140000, 150000]
    log_debug("Audio token samples:")
    for token_id in sample_audio_ids:
        try:
            decoded = tokenizer.decode([token_id])
            log_debug(f"  ID {token_id} -> '{decoded}'")
        except Exception as e:
            log_debug(f"  ID {token_id} -> ERROR: {e}")

    log_debug("=" * 60)

# =============================================================================
# AUDIO DECODING
# =============================================================================
def decode_vocalino_audio(text_content, snac_model_ref, debug=False):
    """Decode audio tokens from completion text."""
    token_strings = re.findall(r"<custom_token_(\d+)>", text_content)

    if len(token_strings) == 0:
        return None

    # <custom_token_X> mapping:
    # - Tokenizer decodes token_id as <custom_token_{token_id - 128256}>
    # - So <custom_token_X> means token_id = X + 128256
    # - Audio tokens have token_id = 128266 + audio_index
    # - Therefore: audio_index = token_id - 128266 = (X + 128256) - 128266 = X - 10
    #
    # Example: <custom_token_2502> → token_id = 130758 → audio_index = 2492

    # Convert custom_token numbers to actual audio indices (0 to 28671)
    token_ids = [int(t) - 10 for t in token_strings]

    if debug and LOCAL_RANK == 0:
        log_debug(f"Token ID range (after -10 adjust): min={min(token_ids)}, max={max(token_ids)}")
        log_debug(f"Expected range: 0 to {AUDIO_TOKEN_END}")

    # Filter valid IDs
    valid_ids = [t for t in token_ids if 0 <= t < AUDIO_TOKEN_END]

    if debug and LOCAL_RANK == 0:
        log_debug(f"Valid IDs after filter: {len(valid_ids)} / {len(token_ids)}")

    valid_ids = valid_ids[:(len(valid_ids) // 7) * 7]

    if len(valid_ids) < 7:
        return None

    # De-interleave into 3 SNAC layers
    layer_1, layer_2, layer_3 = [], [], []

    for i in range(len(valid_ids) // 7):
        base = 7 * i

        l1_val = valid_ids[base] - LAYER_OFFSETS[0]
        l2_val_a = valid_ids[base + 1] - LAYER_OFFSETS[1]
        l3_val_a = valid_ids[base + 2] - LAYER_OFFSETS[2]
        l3_val_b = valid_ids[base + 3] - LAYER_OFFSETS[3]
        l2_val_b = valid_ids[base + 4] - LAYER_OFFSETS[4]
        l3_val_c = valid_ids[base + 5] - LAYER_OFFSETS[5]
        l3_val_d = valid_ids[base + 6] - LAYER_OFFSETS[6]

        all_vals = [l1_val, l2_val_a, l2_val_b, l3_val_a, l3_val_b, l3_val_c, l3_val_d]

        # Debug first frame
        if debug and i == 0 and LOCAL_RANK == 0:
            log_debug(f"Frame 0 raw IDs: {valid_ids[base:base+7]}")
            log_debug(f"Frame 0 after offset: {all_vals}")

        if not all(0 <= v < SNAC_VOCAB_SIZE for v in all_vals):
            if debug and LOCAL_RANK == 0 and i < 3:
                log_debug(f"Skipping frame {i}: values out of range {all_vals}")
            continue

        layer_1.append(l1_val)
        layer_2.append(l2_val_a)
        layer_2.append(l2_val_b)
        layer_3.append(l3_val_a)
        layer_3.append(l3_val_b)
        layer_3.append(l3_val_c)
        layer_3.append(l3_val_d)

    if not layer_1:
        return None

    if debug and LOCAL_RANK == 0:
        log_debug(f"SNAC decode: {len(layer_1)} frames, layers: {len(layer_1)}/{len(layer_2)}/{len(layer_3)}")

    try:
        with torch.no_grad():
            z0 = torch.tensor(layer_1, dtype=torch.long, device=SNAC_DEVICE).unsqueeze(0)
            z1 = torch.tensor(layer_2, dtype=torch.long, device=SNAC_DEVICE).unsqueeze(0)
            z2 = torch.tensor(layer_3, dtype=torch.long, device=SNAC_DEVICE).unsqueeze(0)
            audio = snac_model_ref.decode([z0, z1, z2])
            result = audio.squeeze().cpu().numpy()
            if debug and LOCAL_RANK == 0:
                log_debug(f"SNAC decode success: {len(result)} samples, {len(result)/24000:.2f}s")
            return result
    except Exception as e:
        if LOCAL_RANK == 0:
            log_debug(f"SNAC decode error: {e}")
            import traceback
            traceback.print_exc()
        return None

# =============================================================================
# REWARDS
# =============================================================================
GLOBAL_AUDIO_CACHE = []

def wer_reward(prompts, completions, expected_text, **kwargs):
    GLOBAL_AUDIO_CACHE.clear()
    scores = []

    if LOCAL_RANK == 0:
        log_debug(f"Target Text: '{expected_text[0][:30]}...'")
        # Debug: show what prompt was actually used
        log_debug(f"Prompts type: {type(prompts)}, len: {len(prompts) if prompts else 0}")
        if prompts:
            prompt_preview = prompts[0] if isinstance(prompts[0], str) else str(prompts[0])
            log_debug(f"Prompt preview: {prompt_preview[:200]}...")
            log_debug(f"Prompt ends with: ...{prompt_preview[-100:]}")
        log_debug(f"Completions type: {type(completions)}, len: {len(completions)}")

    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = str(completion)

        if LOCAL_RANK == 0 and i == 0:
            log_debug(f"Completion length: {len(content)} chars")
            log_debug(f"Completion preview: {content[:200]}...")
            log_debug(f"Completion ends with: ...{content[-100:]}")
            token_strings = re.findall(r"<custom_token_(\d+)>", content)
            log_debug(f"Found {len(token_strings)} audio tokens")
            # Check for END_OF_SPEECH (128258 -> custom_token_2)
            if "<custom_token_2>" in content:
                log_debug("END_OF_SPEECH found in completion!")
            else:
                log_debug("WARNING: No END_OF_SPEECH token - model hit max length")

        audio_np = decode_vocalino_audio(content, snac_model, debug=(i == 0))
        GLOBAL_AUDIO_CACHE.append(audio_np)

        if audio_np is None:
            if len(content) > 200:
                scores.append(-2.0)
            else:
                scores.append(-1.0)
            continue

        try:
            transcribed = asr_pipe(audio_np, return_timestamps=True)["text"].lower().strip()
            target = expected_text[i].lower().strip() if isinstance(expected_text, list) else expected_text[0].lower().strip()
            wer_val = compute_wer(target, transcribed)
            score = max(-1.0, 1.0 - wer_val)

            if LOCAL_RANK == 0 and i == 0:
                log_debug(f"WER: {wer_val:.2f} | Score: {score:.2f} | Tx: '{transcribed[:30]}...'")

            scores.append(score)
        except Exception as e:
            if LOCAL_RANK == 0:
                log_debug(f"ASR error: {e}")
            scores.append(-1.0)

    # Cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return scores

def clap_reward(prompts, completions, caption, **kwargs):
    if LOCAL_RANK != 0:
        return [0.0] * len(completions)

    target_cap = caption[0] if isinstance(caption, list) else caption

    scores = []
    for audio_np in GLOBAL_AUDIO_CACHE:
        if audio_np is None:
            scores.append(0.0)
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_np, SAMPLE_RATE)
                temp_path = f.name

            results = majestrino_tagger.tag(temp_path, threshold=95.0, top_n_per_category=3)
            os.unlink(temp_path)

            tags = ', '.join([r['label'] for r in results if r['category'] not in ['Content Rating']])

            if not tags:
                scores.append(-0.5)
                continue

            expected_inputs = gte_tokenizer(target_cap, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                expected_emb = F.normalize(gte_model(**expected_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)

            tags_inputs = gte_tokenizer(tags, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                tags_emb = F.normalize(gte_model(**tags_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)

            similarity = torch.matmul(expected_emb, tags_emb.t()).item()
            scores.append(similarity / 10.0)

        except Exception:
            scores.append(0.0)

    return scores

# =============================================================================
# PROMPT FORMATTING - Match working_inference.py EXACTLY
# =============================================================================
def format_vocalino_prompt(example, tokenizer):
    """Build prompt matching working_inference.py format exactly.

    Working format:
    [START_OF_HUMAN] + encode("Text: {caption}. {text}") + [128009, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    """
    caption = example['caption'].strip().rstrip('.')  # Remove trailing period
    text = example['text']

    # Match working_inference.py: "{caption}. {text}" or just "{text}"
    if caption:
        prompt_content = f"{caption}. {text}"
    else:
        prompt_content = text

    # Encode with "Text: " prefix like working_inference.py line 184
    content_ids = tokenizer.encode("Text: " + prompt_content, add_special_tokens=False)

    # Build prompt with EXACT sequence from working_inference.py line 185:
    # [START_OF_HUMAN] + content + [128009, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    # Note: 128009 = END_OF_TEXT = <|eot_id|> - THIS WAS MISSING!
    prompt_ids = [START_OF_HUMAN] + content_ids + [END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]

    # Decode to text for TRL
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)

    # Verify round-trip (only on first call)
    if not hasattr(format_vocalino_prompt, '_checked'):
        format_vocalino_prompt._checked = True
        re_encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
        if prompt_ids != re_encoded:
            log_debug(f"[WARNING] Round-trip mismatch!")
            log_debug(f"  Original IDs (last 10): {prompt_ids[-10:]}")
            log_debug(f"  Re-encoded (last 10): {re_encoded[-10:]}")
            log_debug(f"  Prompt text (last 100 chars): ...{prompt_text[-100:]}")
        else:
            log_debug(f"[OK] Round-trip successful! Last 5 IDs: {prompt_ids[-5:]}")

        # Show exact prompt structure
        log_debug(f"  Prompt IDs: [{prompt_ids[0]}] + text + {prompt_ids[-4:]}")
        log_debug(f"  Expected:   [START_OF_HUMAN=128259] + text + [EOT=128009, END_OF_HUMAN=128260, START_OF_AI=128261, START_OF_SPEECH=128257]")
        log_debug(f"  Prompt ends with: ...{prompt_text[-80:]}")

    return {
        "prompt": prompt_text,
        "expected_text": text,
        "caption": caption,
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Load tokenizer without modification
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    # Set EOS to END_OF_SPEECH so generation stops at speech end
    tokenizer.eos_token_id = END_OF_SPEECH
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = END_OF_TEXT

    tokenizer.padding_side = "left"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]

    # Run diagnostics
    diagnose_tokenizer(tokenizer)

    # Load model without modification
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.bfloat16)

    # Set EOS on model config for vLLM to respect
    model.config.eos_token_id = END_OF_SPEECH

    # Load helper models
    log_debug(f"Loading SNAC on {SNAC_DEVICE}...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(SNAC_DEVICE)

    log_debug("Loading ASR pipeline (whisper-tiny for memory efficiency)...")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=LOCAL_RANK,
        generate_kwargs={"language": "en", "task": "transcribe"}
    )

    gte_model = None
    gte_tokenizer = None
    majestrino_tagger = None

    if LOCAL_RANK == 0:
        log_debug("Loading GTE model...")
        gte_model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(DEVICE).eval()
        gte_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")

        log_debug("Loading Majestrino tagger...")
        majestrino_tagger = MajestrinoTagger.from_pretrained()
        majestrino_tagger.load_tags()

    # Test prompt formatting
    sample = {"caption": "test caption", "text": "Hello world"}
    sample_prompt = format_vocalino_prompt(sample, tokenizer)
    log_debug(f"Sample prompt: {sample_prompt['prompt']}")

    training_args = GRPOConfig(
        output_dir="vocalino_0.11_grpo_new",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # Reduced from 4
        num_generations=4,
        max_prompt_length=512,  # Reduced from 1024
        max_completion_length=1536,  # Reduced from 2048
        temperature=0.8,
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,

        # vLLM for fast generation
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,  # Reduced from 0.3
        vllm_max_model_length=2560,  # Reduced from 3072
        vllm_tensor_parallel_size=1,
        vllm_enable_sleep_mode=True,

        # Generation params
        top_p=0.95,
        generation_kwargs={"stop_token_ids": [END_OF_SPEECH, END_OF_TEXT]},
    )

    dataset = load_dataset(DATASET_NAME, split="train")
    log_debug(f"Dataset columns: {dataset.column_names}")
    log_debug(f"Dataset size before filter: {len(dataset)}")

    # Filter to English only to avoid language confusion
    if "language" in dataset.column_names:
        dataset = dataset.filter(lambda x: x.get("language", "en") == "en")
        log_debug(f"Filtered to English: {len(dataset)} samples")
    else:
        log_debug("No 'language' column found - using all samples")

    dataset = dataset.map(lambda x: format_vocalino_prompt(x, tokenizer))

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[wer_reward, clap_reward],
        args=training_args,
        train_dataset=dataset,
    )

    log_debug("Starting Training...")
    trainer.train()

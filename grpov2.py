import sys
import os
import numpy as np
import tempfile
import soundfile as sf
from majestrino_tagger import MajestrinoTagger

# =============================================================================
# ENVIRONMENT
# =============================================================================
sys.modules["vllm"] = None  # Disable vLLM - causes issues with FSDP
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import load_dataset
from snac import SNAC
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    WhisperModel,
    WhisperFeatureExtractor,
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
MAX_SEQ_LENGTH = 4096
SAMPLE_RATE = 24000

# Device config
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"
SNAC_DEVICE = "cpu"

# =============================================================================
# VOCALINO TOKENS
# =============================================================================
TOKEN_OFFSET_BASE = 128266
SNAC_VOCAB_SIZE = 4096
# TRL decodes token IDs by subtracting TOKEN_OFFSET_BASE
# So <custom_token_X> where X is in range 0 to 7*4096 = 28672
LAYER_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]  # 7 layers
AUDIO_TOKEN_END = 7 * SNAC_VOCAB_SIZE  # 28672 (decoded range)

# Explicit Strings for mapping
TOK_START_SPEECH = "<|startofspeech|>"
TOK_END_SPEECH = "<|endofspeech|>"
TOK_START_AI = "<|startofai|>"
TOK_START_HUMAN = "<|startofhuman|>"
TOK_END_HUMAN = "<|endofhuman|>"
TOK_END_TEXT = "<|endoftext|>"

# =============================================================================
# LOGGING
# =============================================================================
def log_debug(msg):
    if LOCAL_RANK == 0:
        print(f"[DEBUG] {msg}", flush=True)

# =============================================================================
# MODEL PREPARATION
# =============================================================================
def prepare_local_model():
    marker_file = os.path.join(LOCAL_MODEL_PATH, ".ready")
    if os.path.exists(marker_file): return
    if LOCAL_RANK == 0:
        log_debug("Preparing local model...")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model_tmp = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16)
        tokenizer_tmp = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        tokenizer_tmp.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        model_tmp.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer_tmp.save_pretrained(LOCAL_MODEL_PATH)
        with open(marker_file, "w") as f: f.write("ready")
    else:
        while not os.path.exists(marker_file): time.sleep(1)

prepare_local_model()

# =============================================================================
# AUDIO DECODING
# =============================================================================
def sanitize_indices(lst):
    return [max(0, min(SNAC_VOCAB_SIZE - 1, x)) for x in lst]

def decode_vocalino_audio(text_content, snac_model_ref, debug=False):
    """Decode audio tokens from completion text.

    Note: TRL decodes token IDs by subtracting TOKEN_OFFSET_BASE, so
    <custom_token_X> contains raw SNAC indices (0-12287), not vocab IDs.
    """
    token_strings = re.findall(r"<custom_token_(\d+)>", text_content)

    if len(token_strings) == 0:
        return None

    token_ids = [int(t) for t in token_strings]

    if debug and LOCAL_RANK == 0:
        log_debug(f"Token ID range: min={min(token_ids)}, max={max(token_ids)}")
        log_debug(f"Expected range: 0 to {AUDIO_TOKEN_END}")

    # Filter valid IDs (already decoded, so range is 0 to 12288)
    valid_ids = [t for t in token_ids if 0 <= t < AUDIO_TOKEN_END]

    if debug and LOCAL_RANK == 0:
        log_debug(f"Valid IDs after filter: {len(valid_ids)} / {len(token_ids)}")

    valid_ids = valid_ids[:(len(valid_ids) // 7) * 7]

    if len(valid_ids) < 7:
        if debug and LOCAL_RANK == 0:
            log_debug(f"Not enough valid IDs: {len(valid_ids)}")
        return None

    # De-interleave into 3 SNAC layers (matching train_laion.py)
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
        if not all(0 <= v < SNAC_VOCAB_SIZE for v in all_vals):
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

    try:
        with torch.no_grad():
            z0 = torch.tensor(layer_1, dtype=torch.long, device=SNAC_DEVICE).unsqueeze(0)
            z1 = torch.tensor(layer_2, dtype=torch.long, device=SNAC_DEVICE).unsqueeze(0)
            z2 = torch.tensor(layer_3, dtype=torch.long, device=SNAC_DEVICE).unsqueeze(0)
            audio = snac_model_ref.decode([z0, z1, z2])
            return audio.squeeze().cpu().numpy()
    except Exception as e:
        if LOCAL_RANK == 0:
            log_debug(f"SNAC decode error: {e}")
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

    for i, completion in enumerate(completions):
        if isinstance(completion, list): content = completion[0]["content"]
        else: content = str(completion)

        # Debug: log what the model is generating
        if LOCAL_RANK == 0 and i == 0:
            log_debug(f"Completion length: {len(content)} chars")
            log_debug(f"Completion preview: {content[:200]}...")
            token_strings = re.findall(r"<custom_token_(\d+)>", content)
            log_debug(f"Found {len(token_strings)} audio tokens")

        audio_np = decode_vocalino_audio(content, snac_model, debug=(i == 0))
        GLOBAL_AUDIO_CACHE.append(audio_np)

        if audio_np is None:
            # Massive penalty for text hallucination
            if len(content) > 200:
                scores.append(-2.0)
            else:
                scores.append(-1.0)
            continue

        try:
            transcribed = asr_pipe(audio_np)["text"].lower().strip()
            target = expected_text[i].lower().strip() if isinstance(expected_text, list) else expected_text[0].lower().strip()
            wer_val = compute_wer(target, transcribed)
            score = max(-1.0, 1.0 - wer_val)

            if LOCAL_RANK == 0 and i == 0:
                log_debug(f"Success! WER: {wer_val:.2f} | Score: {score:.2f} | Tx: '{transcribed[:30]}...'")

            scores.append(score)
        except Exception:
            scores.append(-1.0)
    return scores

def clap_reward(prompts, completions, caption, **kwargs):
    """CLAP reward using Majestrino tagger + GTE embeddings."""
    # Only rank 0 has these models loaded
    if LOCAL_RANK != 0:
        return [0.0] * len(completions)

    target_cap = caption[0] if isinstance(caption, list) else caption

    scores = []
    for audio_np in GLOBAL_AUDIO_CACHE:
        if audio_np is None:
            scores.append(0.0)
            continue
        try:
            # Save temp audio file for tagger
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_np, SAMPLE_RATE)
                temp_path = f.name

            # Tag the audio
            results = majestrino_tagger.tag(temp_path, threshold=95.0, top_n_per_category=3)
            os.unlink(temp_path)

            # Get tags (excluding Content Rating)
            tags = ', '.join([r['label'] for r in results if r['category'] not in ['Content Rating']])

            if not tags:
                scores.append(-0.5)
                continue

            # Get embeddings for expected prompt
            expected_inputs = gte_tokenizer(target_cap, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                expected_emb = F.normalize(gte_model(**expected_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)

            # Get embeddings for tags
            tags_inputs = gte_tokenizer(tags, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                tags_emb = F.normalize(gte_model(**tags_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)

            # Compute similarity (ranges from -1 to 1), scale down
            similarity = torch.matmul(expected_emb, tags_emb.t()).item()
            scores.append(similarity / 10.0)

        except Exception:
            scores.append(0.0)

    return scores

# =============================================================================
# TOKENIZER PATCHING (CRITICAL FIX)
# =============================================================================
def patch_tokenizer(tokenizer):
    """
    Force-register special tokens with specific strings so we can construct
    prompts that survive the ID->String->ID roundtrip.
    """
    special_tokens = [
        TOK_START_SPEECH, TOK_END_SPEECH,
        TOK_START_AI, TOK_START_HUMAN, TOK_END_HUMAN
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    return tokenizer

def format_vocalino_prompt(example):
    caption = example['caption']
    text = example['text']

    # Construct prompt using the EXPLICIT STRINGS we registered
    prompt_str = (
        f"{TOK_START_HUMAN}"
        f"<start_of_caption>{caption}<end_of_caption>{text}"
        f"{TOK_END_HUMAN}"
        f"{TOK_START_AI}{TOK_START_SPEECH}"
    )

    return {
        "prompt": prompt_str,
        "expected_text": text,
        "caption": caption,
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # 1. Load Tokenizer & Patch
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    patch_tokenizer(tokenizer)

    # Fix padding
    if tokenizer.eos_token_id is None: tokenizer.eos_token_id = 128009
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 128009
    tokenizer.padding_side = "left"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]

    # 2. Load Model & Resize Embeddings for new tokens
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Load Helper Models
    log_debug(f"Loading SNAC on {SNAC_DEVICE}...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(SNAC_DEVICE)

    log_debug("Loading ASR pipeline...")
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=LOCAL_RANK)

    # GTE and Majestrino only on rank 0
    gte_model = None
    gte_tokenizer = None
    majestrino_tagger = None

    if LOCAL_RANK == 0:
        log_debug("Loading GTE model for similarity...")
        gte_model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(DEVICE).eval()
        gte_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")

        log_debug("Loading Majestrino tagger...")
        majestrino_tagger = MajestrinoTagger.from_pretrained()
        majestrino_tagger.load_tags()

    training_args = GRPOConfig(
        output_dir="vocalino_0.11_grpo_new",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=2048,
        temperature=0.8,
        bf16=True,
        use_vllm=False,
        report_to="wandb",
        remove_unused_columns=False,
    )

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(format_vocalino_prompt)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[wer_reward, clap_reward],
        args=training_args,
        train_dataset=dataset,
    )

    log_debug("Starting Training...")
    trainer.train()

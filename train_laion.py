import json
import os
import re
import tempfile
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import soundfile as sf
from datasets import Audio, load_dataset
from snac import SNAC
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    pipeline,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from trl import GRPOConfig, GRPOTrainer
from jiwer import wer as compute_wer, cer as compute_cer
from audiobox_aesthetics.infer import initialize_predictor
from majestrino_tagger import MajestrinoTagger

# =============================================================================
# CONFIG
# =============================================================================
import time

BASE_MODEL_NAME = "ChristophSchuhmann/Vocalino_0.11_alpha"
LOCAL_MODEL_PATH = "./vocalino_ct"  # Local copy with chat template
DATASET_NAME = "mrfakename/emoact_prompts_with_language"
MAX_SEQ_LENGTH = 4096
SAMPLE_RATE = 24000

# Multi-GPU support: use LOCAL_RANK to determine device
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"
print(f"[Rank {LOCAL_RANK}] Using device: {DEVICE}")

# =============================================================================
# PREPARE LOCAL MODEL WITH CHAT TEMPLATE (for vLLM compatibility)
# =============================================================================
def prepare_local_model():
    """Save model + tokenizer with chat template locally. Only rank 0 does this."""
    marker_file = os.path.join(LOCAL_MODEL_PATH, ".ready")

    if os.path.exists(marker_file):
        print(f"[Rank {LOCAL_RANK}] Local model already exists at {LOCAL_MODEL_PATH}")
        return

    if LOCAL_RANK == 0:
        print(f"[Rank 0] Preparing local model with chat template...")

        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_tmp = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16)
        tokenizer_tmp = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # Set chat template
        tokenizer_tmp.chat_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""

        # Save locally
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model_tmp.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer_tmp.save_pretrained(LOCAL_MODEL_PATH)

        # Create marker file to signal completion
        with open(marker_file, "w") as f:
            f.write("ready")

        print(f"[Rank 0] Saved model with chat template to {LOCAL_MODEL_PATH}")
        del model_tmp, tokenizer_tmp
        torch.cuda.empty_cache()
    else:
        # Wait for rank 0 to finish
        print(f"[Rank {LOCAL_RANK}] Waiting for rank 0 to prepare local model...")
        while not os.path.exists(marker_file):
            time.sleep(1)
        print(f"[Rank {LOCAL_RANK}] Local model ready")

prepare_local_model()
MODEL_NAME = LOCAL_MODEL_PATH  # Use local model for training

# =============================================================================
# SPECIAL TOKENS (LAION format)
# =============================================================================
TOKEN_OFFSET_BASE = 128266
LAYER_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]

START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
END_OF_TEXT = 128009

# =============================================================================
# LOAD BASE MODEL (with FSDP support)
# =============================================================================
print("Loading model for FSDP training...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    # Don't use device_map with FSDP - let FSDP handle distribution
    use_cache=False,  # Required for gradient checkpointing with FSDP
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Ensure tokenizer has proper special tokens
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = END_OF_TEXT
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Disable token_type_ids (not used by this model) - force override
tokenizer.model_input_names = ['input_ids', 'attention_mask']

# Patch __call__ to remove token_type_ids
_original_tokenizer_call = tokenizer.__class__.__call__
def _patched_call(self, *args, **kwargs):
    result = _original_tokenizer_call(self, *args, **kwargs)
    if hasattr(result, 'pop'):
        result.pop('token_type_ids', None)
    elif isinstance(result, dict):
        result.pop('token_type_ids', None)
    return result
tokenizer.__class__.__call__ = _patched_call

# Chat template is already in the local model (saved by prepare_local_model)

# =============================================================================
# LOAD SNAC
# =============================================================================
print("Loading SNAC...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)

# =============================================================================
# LOAD REWARD MODELS
# =============================================================================
print("Loading reward models...")

# ASR for WER - use large model for best quality (H100s can handle it)
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device=LOCAL_RANK,
    torch_dtype=torch.bfloat16,
)

# Majestrino Tagger and Audiobox only on rank 0 (they use cuda:0)
majestrino_tagger = None
audiobox_predictor = None
gte_tokenizer = None
gte_model = None

if LOCAL_RANK == 0:
    print(f"[Rank 0] Loading MajestrinoTagger...")
    majestrino_tagger = MajestrinoTagger.from_pretrained()
    majestrino_tagger.load_tags()
    print(f"[Rank 0] MajestrinoTagger loaded")

    # GTE text encoder for computing similarity between tags and caption
    print(f"[Rank 0] Loading GTE text encoder...")
    gte_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
    gte_model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(DEVICE).eval()
    print(f"[Rank 0] GTE loaded")

    # Audiobox Aesthetics - always runs on cuda:0 (hardcoded in library)
    print(f"[Rank 0] Loading Audiobox...")
    audiobox_predictor = initialize_predictor()
    print(f"[Rank 0] Audiobox loaded")

# =============================================================================
# AUDIO ENCODING (for voice cloning reference)
# =============================================================================
SNAC_VOCAB_SIZE = 4096
AUDIO_TOKEN_START = TOKEN_OFFSET_BASE
AUDIO_TOKEN_END = AUDIO_TOKEN_START + (7 * SNAC_VOCAB_SIZE)

def encode_audio_to_tokens(waveform, orig_sr):
    """Encode audio waveform to SNAC tokens for voice cloning reference."""
    waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
    
    if orig_sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    waveform = waveform.unsqueeze(0).to(DEVICE)
    
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    
    c0 = codes[0].cpu().numpy()[0]
    c1 = codes[1].cpu().numpy()[0]
    c2 = codes[2].cpu().numpy()[0]
    
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.extend([
            c0[i] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[0],
            c1[2*i] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[1],
            c2[4*i] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[2],
            c2[4*i+1] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[3],
            c1[2*i+1] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[4],
            c2[4*i+2] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[5],
            c2[4*i+3] + TOKEN_OFFSET_BASE + LAYER_OFFSETS[6]
        ])
    
    return all_codes

# =============================================================================
# AUDIO DECODING (for reward computation)
# =============================================================================
def is_valid_audio_token(token_id):
    return AUDIO_TOKEN_START <= token_id < AUDIO_TOKEN_END

def count_invalid_tokens(token_ids):
    return sum(1 for t in token_ids if not is_valid_audio_token(t))

def filter_valid_audio_tokens(token_ids):
    valid = [t for t in token_ids if is_valid_audio_token(t)]
    valid = valid[:(len(valid) // 7) * 7]
    return valid

def prepare_snac_codes(token_ids, device=None):
    if device is None:
        device = DEVICE
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    token_ids = filter_valid_audio_tokens(token_ids)
    
    if len(token_ids) < 7:
        empty = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)
        return [empty.clone(), empty.clone(), empty.clone()]
    
    layer_1, layer_2, layer_3 = [], [], []
    
    for i in range(len(token_ids) // 7):
        base = 7 * i
        
        l1_val = token_ids[base] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[0]
        l2_val_a = token_ids[base + 1] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[1]
        l3_val_a = token_ids[base + 2] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[2]
        l3_val_b = token_ids[base + 3] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[3]
        l2_val_b = token_ids[base + 4] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[4]
        l3_val_c = token_ids[base + 5] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[5]
        l3_val_d = token_ids[base + 6] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[6]
        
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
        empty = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)
        return [empty.clone(), empty.clone(), empty.clone()]
    
    return [
        torch.tensor(layer_1, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.long, device=device).unsqueeze(0),
    ]

def decode_tokens_to_audio(token_ids, chunk_size=700):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    if not token_ids:
        return None
    
    token_ids = filter_valid_audio_tokens(token_ids)
    
    if len(token_ids) < 7:
        return None
    
    segments = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        codes = prepare_snac_codes(chunk, device=DEVICE)
        
        if codes[0].numel() == 0:
            continue
        
        try:
            with torch.no_grad():
                audio = snac_model.decode(codes)
                if audio.ndim == 2:
                    audio = audio.unsqueeze(1)
                segments.append(audio.cpu())
        except Exception as e:
            print(f"[Rank {LOCAL_RANK}] SNAC decode error: {e}")
            continue
    
    if not segments:
        return None
    
    return torch.cat(segments, dim=2)

# =============================================================================
# DATASET PREPARATION
# =============================================================================
import random

print("Loading datasets...")
dataset = load_dataset(DATASET_NAME, split="train")

# Load voice prompts dataset for random voice cloning references
import pickle

VOICE_CACHE_PATH = "voice_prompts_cache.pkl"
VOICE_EMOTION_GRID_PATH = "voice_emotion_grid.jsonl"
MAX_REF_SECONDS = 1.5  # Shorter reference = faster training
MAX_REF_SAMPLES = int(SAMPLE_RATE * MAX_REF_SECONDS)
MAX_REF_TOKENS = 175  # ~1.5 seconds of audio
ENCODE_BATCH_SIZE = 64

# Load voice emotion grid for caption augmentation
VOICE_EMOTION_DESCRIPTIONS = []
if os.path.exists(VOICE_EMOTION_GRID_PATH):
    print(f"[Rank {LOCAL_RANK}] Loading voice emotion grid from {VOICE_EMOTION_GRID_PATH}...")
    with open(VOICE_EMOTION_GRID_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            VOICE_EMOTION_DESCRIPTIONS.append(record["description"])
    print(f"[Rank {LOCAL_RANK}] Loaded {len(VOICE_EMOTION_DESCRIPTIONS)} voice emotion descriptions")
else:
    print(f"[Rank {LOCAL_RANK}] Warning: {VOICE_EMOTION_GRID_PATH} not found. Run var.py first to generate it.")

# Load voice prompts cache (must be generated first with generate_voice_cache.py)
VOICE_PROMPT_TOKENS = []
if os.path.exists(VOICE_CACHE_PATH):
    with open(VOICE_CACHE_PATH, "rb") as f:
        VOICE_PROMPT_TOKENS = pickle.load(f)
    print(f"[Rank {LOCAL_RANK}] Loaded {len(VOICE_PROMPT_TOKENS)} cached voice prompts")
else:
    raise RuntimeError(
        f"Voice cache not found at {VOICE_CACHE_PATH}. "
        "Run `python generate_voice_cache.py` first to generate it."
    )

def process_example(example):
    """
    Create prompt matching sft.py format:
    <START_OF_HUMAN><start_of_caption>{caption}<end_of_caption>{text}<END_OF_TEXT><END_OF_HUMAN><START_OF_AI><START_OF_SPEECH>{ref_audio_tokens}
    """
    # Pick a random voice prompt for voice cloning (50% of the time)
    if random.random() < 0.5:
        ref_tokens = random.choice(VOICE_PROMPT_TOKENS)
    else:
        ref_tokens = []  # No voice prompt - model learns to generate without cloning

    caption = example.get("caption", "")
    text = example.get("text", "")

    # 25% of the time, replace caption with a random voice emotion description
    if VOICE_EMOTION_DESCRIPTIONS and random.random() < 0.25:
        caption = random.choice(VOICE_EMOTION_DESCRIPTIONS)

    # Tokenize the text part: <start_of_caption>{caption}<end_of_caption>{text}
    text_content = f"<start_of_caption>{caption}<end_of_caption>{text}"
    text_ids = tokenizer.encode(text_content, add_special_tokens=False)

    # Build full prompt token sequence (model continues from here)
    prompt_ids = (
        [START_OF_HUMAN]
        + text_ids
        + [END_OF_TEXT, END_OF_HUMAN]
        + [START_OF_AI, START_OF_SPEECH]
        + ref_tokens  # Reference audio for voice cloning
    )

    return {
        "prompt": {"prompt_token_ids": prompt_ids},  # vLLM format for token IDs
        "expected_text": text,  # For WER comparison
        "caption": caption,  # For CLAP comparison
    }

print("Processing dataset...")
print(f"Dataset columns: {dataset.column_names}")
print(f"Dataset size before processing: {len(dataset)}")

# Check first example
if len(dataset) > 0:
    print(f"First example keys: {dataset[0].keys()}")

dataset = dataset.map(process_example)

print(f"Dataset size after processing: {len(dataset)}")

if len(dataset) == 0:
    raise ValueError("Dataset is empty after processing!")

# =============================================================================
# GLOBAL STATE FOR REWARDS
# =============================================================================
GLOBAL_AUDIO = []
GLOBAL_EXPECTED_TEXT = []

# =============================================================================
# REWARD FUNCTIONS
# =============================================================================
@torch.inference_mode()
def get_gte_embedding(text):
    """Get GTE text embedding for similarity computation."""
    inputs = gte_tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    outputs = gte_model(**inputs)
    mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return F.normalize(sum_emb / sum_mask, p=2, dim=1)

def tags_to_string(tags):
    """Convert majestrino tags to a descriptive string."""
    if not tags:
        return ""
    # Sort by probability and concatenate labels
    sorted_tags = sorted(tags, key=lambda x: x['prob'], reverse=True)
    return ", ".join(t['label'] for t in sorted_tags)

INVALID_TOKEN_PENALTY = 0.1

def wer_reward(prompts, completions, expected_text, **kwargs):
    GLOBAL_AUDIO.clear()
    GLOBAL_EXPECTED_TEXT.clear()

    # expected_text is a list (one per completion)
    if not isinstance(expected_text, list):
        expected_text = [expected_text] * len(completions)

    # Debug: log first completion to see what model generates
    if LOCAL_RANK == 0 and len(completions) > 0:
        first_content = completions[0]
        print(f"[DEBUG] First completion ({len(first_content)} chars): {first_content[:200]}...")

    pattern = r"<custom_token_\d+>"
    scores = []

    for idx, completion in enumerate(completions):
        # Get expected text for this completion
        exp_text = expected_text[idx] if idx < len(expected_text) else expected_text[0]
        GLOBAL_EXPECTED_TEXT.append(exp_text)

        content = completion  # TRL passes completions as strings directly
        tokens = re.findall(pattern, content)
        codes = tokenizer.encode("".join(tokens))[1:] if tokens else []

        # Debug: log token extraction
        if LOCAL_RANK == 0 and idx == 0:
            print(f"[DEBUG] Found {len(tokens)} custom_tokens, {len(codes)} codes")

        if not codes:
            scores.append(-5.0)
            GLOBAL_AUDIO.append(None)
            continue

        num_invalid = count_invalid_tokens(codes)
        invalid_penalty = num_invalid * INVALID_TOKEN_PENALTY

        try:
            audio = decode_tokens_to_audio(codes)
            if audio is None:
                scores.append(-5.0 - invalid_penalty)
                GLOBAL_AUDIO.append(None)
                continue

            # Store on CPU to avoid cross-device issues in subsequent reward functions
            GLOBAL_AUDIO.append(audio.cpu())

            audio_np = audio.squeeze().cpu().numpy()
            result = asr_pipe(audio_np, return_timestamps=True)
            transcribed = result["text"].strip()

            # Compute both WER and CER for better accuracy signal
            ref_lower = exp_text.lower().strip()
            hyp_lower = transcribed.lower().strip()

            # Debug: log transcription for first completion
            if LOCAL_RANK == 0 and idx == 0:
                print(f"[DEBUG] Expected: {ref_lower[:100]}...")
                print(f"[DEBUG] Transcribed: {hyp_lower[:100]}...")

            if not hyp_lower:
                # Empty transcription
                scores.append(-3.0 - invalid_penalty)
                continue

            wer_score = compute_wer(ref_lower, hyp_lower)
            cer_score = compute_cer(ref_lower, hyp_lower)

            # Combine WER (70%) and CER (30%) - CER rewards partial word matches
            combined_error = 0.7 * wer_score + 0.3 * cer_score
            accuracy = max(0, 1 - combined_error)
            final_score = accuracy - invalid_penalty
            scores.append(final_score)

        except Exception as e:
            print(f"[Rank {LOCAL_RANK}] WER error: {e}")
            scores.append(-5.0 - invalid_penalty)
            GLOBAL_AUDIO.append(None)

    return scores

def majestrino_reward(prompts, completions, caption, **kwargs):
    """Compute caption similarity using MajestrinoTagger + GTE embeddings."""
    # MajestrinoTagger only works on cuda:0, so only rank 0 computes this
    if LOCAL_RANK != 0:
        return [0.0] * len(completions)

    # caption is a list (one per completion)
    if not isinstance(caption, list):
        caption = [caption] * len(completions)

    scores = []
    for idx in range(len(completions)):
        cap_text = caption[idx] if idx < len(caption) else caption[0]
        audio = GLOBAL_AUDIO[idx] if idx < len(GLOBAL_AUDIO) else None

        if not cap_text or audio is None:
            scores.append(-1.0)
            continue

        try:
            # Save audio to temp file for majestrino
            audio_np = audio.squeeze().cpu().numpy()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_np, SAMPLE_RATE)
                temp_path = f.name

            # Tag audio with majestrino (threshold=95% for high confidence tags)
            tags = majestrino_tagger.tag(temp_path, threshold=95.0, top_n_per_category=5)
            os.unlink(temp_path)  # Clean up temp file

            if not tags:
                scores.append(-0.5)  # No tags detected
                continue

            # Convert tags to string and compute GTE similarity
            tags_str = tags_to_string(tags)
            tags_emb = get_gte_embedding(tags_str)
            caption_emb = get_gte_embedding(cap_text)
            similarity = torch.matmul(tags_emb, caption_emb.t()).item()
            scores.append(similarity)

        except Exception as e:
            print(f"[Rank {LOCAL_RANK}] Majestrino error: {e}")
            scores.append(-1.0)

    return scores

def audiobox_reward(prompts, completions, **kwargs):
    num_completions = len(completions)
    scores = [0.0] * num_completions  # Pre-fill with neutral scores

    # Audiobox only works on cuda:0, so only rank 0 computes this
    if LOCAL_RANK != 0:
        GLOBAL_AUDIO.clear()
        return scores

    valid_audios = [(i, audio) for i, audio in enumerate(GLOBAL_AUDIO) if audio is not None and i < num_completions]

    if not valid_audios:
        GLOBAL_AUDIO.clear()
        return scores

    try:
        # Audiobox predictor always uses cuda:0 - move audio there explicitly
        predictions = audiobox_predictor.forward([
            {"path": audio.squeeze().unsqueeze(0).to("cuda:0"), "sample_rate": 24000}
            for _, audio in valid_audios
        ])

        for i, pred in enumerate(predictions):
            idx = valid_audios[i][0]
            if idx < num_completions:
                pq = pred["PQ"]
                scores[idx] = (pq - 5) / 5

    except Exception as e:
        print(f"[Rank {LOCAL_RANK}] Audiobox error: {e}")

    GLOBAL_AUDIO.clear()
    return scores

# =============================================================================
# TRAINING CONFIG WITH FSDP
# =============================================================================
training_args = GRPOConfig(
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    save_steps=100,
    report_to="wandb",
    output_dir="outputs_laion",
    bf16=True,

    num_generations=2,  # Reduced to save memory
    max_prompt_length=2048,
    max_completion_length=1500,

    # Sampling parameters (same for both vLLM and HF generate)
    temperature=0.8,
    top_p=0.95,
    top_k=50,  # vLLM needs an int, not None
    min_p=0.1,

    # vLLM uses different param names than HF generate
    generation_kwargs={
        "stop_token_ids": [END_OF_SPEECH, tokenizer.eos_token_id],
    },

    # vLLM for fast generation (3-5x speedup)
    use_vllm=True,
    vllm_mode="colocate",  # Share GPU with training (vs "server" mode)
    vllm_gpu_memory_utilization=0.3,  # Reduced for OOM
    vllm_max_model_length=2500,  # Reduced for OOM
    vllm_tensor_parallel_size=1,  # Each GPU runs its own vLLM instance
    vllm_enable_sleep_mode=True,  # Offload vLLM weights during backward pass

    # FSDP Configuration for training
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_forward_prefetch": True,
        "fsdp_use_orig_params": True,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        "fsdp_sync_module_states": True,
    },

    # Gradient checkpointing for memory efficiency with FSDP
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# =============================================================================
# TRAINER
# =============================================================================
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[wer_reward, majestrino_reward, audiobox_reward],
    args=training_args,
    train_dataset=dataset,
)

# =============================================================================
# TRAIN
# =============================================================================
if __name__ == "__main__":
    trainer.train()
    
    model.save_pretrained("outputs_laion/final")
    tokenizer.save_pretrained("outputs_laion/final")

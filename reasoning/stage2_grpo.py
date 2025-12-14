"""
Stage 2: GRPO to improve reasoning and audio quality.

Run this AFTER stage1_sft.py has taught the model the reasoning format.

This script:
1. Loads the SFT'd model from stage 1
2. Uses GRPO with rewards for:
   - Correct reasoning format
   - Good reasoning length
   - Reasoning quality (mentions relevant concepts)
   - WER (speech accuracy)
   - CLAP (audio-text alignment)
   - Audiobox (audio quality)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import Audio, load_dataset
from snac import SNAC
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    WhisperModel,
    WhisperFeatureExtractor,
    pipeline,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from trl import GRPOConfig, GRPOTrainer
from jiwer import wer as compute_wer
from audiobox_aesthetics.infer import initialize_predictor

# =============================================================================
# CONFIG
# =============================================================================
import os
import time

# Load from stage 1 SFT output
SFT_MODEL_PATH = "outputs_reasoning_sft/final"
LOCAL_MODEL_PATH = "./vocalino_reasoning_ct"  # Local copy with chat template
DATASET_NAME = "mrfakename/emoact_prompts_with_language"
MAX_SEQ_LENGTH = 4096
SAMPLE_RATE = 24000

# Multi-GPU support
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = f"cuda:{LOCAL_RANK}"
print(f"[Rank {LOCAL_RANK}] Using device: {DEVICE}")

# =============================================================================
# SPECIAL TOKENS
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

# Reasoning markers
REASONING_START = "<start_of_reasoning>"
REASONING_END = "<end_of_reasoning>"

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

        model_tmp = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH, torch_dtype=torch.bfloat16)
        tokenizer_tmp = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

        # Set chat template
        tokenizer_tmp.chat_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""

        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model_tmp.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer_tmp.save_pretrained(LOCAL_MODEL_PATH)

        with open(marker_file, "w") as f:
            f.write("ready")

        print(f"[Rank 0] Saved model with chat template to {LOCAL_MODEL_PATH}")
        del model_tmp, tokenizer_tmp
        torch.cuda.empty_cache()
    else:
        print(f"[Rank {LOCAL_RANK}] Waiting for rank 0 to prepare local model...")
        while not os.path.exists(marker_file):
            time.sleep(1)
        print(f"[Rank {LOCAL_RANK}] Local model ready")

prepare_local_model()
MODEL_NAME = LOCAL_MODEL_PATH

# =============================================================================
# LOAD BASE MODEL
# =============================================================================
print("Loading model for FSDP training...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = END_OF_TEXT
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.model_input_names = ['input_ids', 'attention_mask']

_original_tokenizer_call = tokenizer.__class__.__call__
def _patched_call(self, *args, **kwargs):
    result = _original_tokenizer_call(self, *args, **kwargs)
    if hasattr(result, 'pop'):
        result.pop('token_type_ids', None)
    elif isinstance(result, dict):
        result.pop('token_type_ids', None)
    return result
tokenizer.__class__.__call__ = _patched_call

REASONING_START_IDS = tokenizer.encode(REASONING_START, add_special_tokens=False)
REASONING_END_IDS = tokenizer.encode(REASONING_END, add_special_tokens=False)

# =============================================================================
# LOAD SNAC
# =============================================================================
print("Loading SNAC...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)

# =============================================================================
# LOAD REWARD MODELS
# =============================================================================
print("Loading reward models...")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device=LOCAL_RANK,
    torch_dtype=torch.bfloat16,
)

class WhisperClapModel(nn.Module):
    def __init__(self, whisper_name):
        super().__init__()
        self.audio_encoder = WhisperModel.from_pretrained(whisper_name).encoder
        self.projector = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Linear(2048, 768)
        )

    def forward(self, input_features):
        outputs = self.audio_encoder(input_features)
        rep = outputs.last_hidden_state.mean(dim=1)
        emb = self.projector(rep)
        return F.normalize(emb, p=2, dim=1)

clap_audio_model = WhisperClapModel("openai/whisper-small").to(DEVICE)
clap_weights = hf_hub_download(repo_id="laion/whisper-clap-version-0.1", filename="model.safetensors")
clap_state = load_file(clap_weights)
clap_state = {k.replace("model.", ""): v for k, v in clap_state.items()}
clap_audio_model.load_state_dict(clap_state, strict=False)
clap_audio_model.eval()

clap_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
clap_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
clap_text_model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(DEVICE).eval()

audiobox_predictor = initialize_predictor()

RESAMPLER_24_TO_16 = T.Resample(orig_freq=24000, new_freq=16000).to(DEVICE)

# =============================================================================
# AUDIO DECODING
# =============================================================================
SNAC_VOCAB_SIZE = 4096
AUDIO_TOKEN_START = TOKEN_OFFSET_BASE
AUDIO_TOKEN_END = AUDIO_TOKEN_START + (7 * SNAC_VOCAB_SIZE)

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
        layer_2.extend([l2_val_a, l2_val_b])
        layer_3.extend([l3_val_a, l3_val_b, l3_val_c, l3_val_d])

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
import pickle

print("Loading datasets...")
dataset = load_dataset(DATASET_NAME, split="train")

# Load voice prompts cache
VOICE_CACHE_PATH = "voice_prompts_cache.pkl"

if os.path.exists(VOICE_CACHE_PATH):
    with open(VOICE_CACHE_PATH, "rb") as f:
        VOICE_PROMPT_TOKENS = pickle.load(f)
    print(f"[Rank {LOCAL_RANK}] Loaded {len(VOICE_PROMPT_TOKENS)} cached voice prompts")
else:
    raise ValueError(f"Voice cache not found at {VOICE_CACHE_PATH}. Run train_laion.py first to create it.")

def process_example(example):
    """
    Create prompt for reasoning TTS.
    Model will generate: reasoning + audio tokens
    """
    ref_tokens = random.choice(VOICE_PROMPT_TOKENS)

    caption = example.get("caption", "")
    text = example.get("text", "")

    text_content = f"<start_of_caption>{caption}<end_of_caption>{text}"
    text_ids = tokenizer.encode(text_content, add_special_tokens=False)

    # Prompt ends with <start_of_reasoning>, model generates from there
    prompt_ids = (
        [START_OF_HUMAN]
        + text_ids
        + [END_OF_TEXT, END_OF_HUMAN]
        + [START_OF_AI]
        + REASONING_START_IDS
    )

    return {
        "prompt": prompt_ids,
        "expected_text": text,
        "caption": caption,
        "ref_tokens": ref_tokens,
    }

print("Processing dataset...")
dataset = dataset.map(process_example)
print(f"Dataset size: {len(dataset)}")

# =============================================================================
# GLOBAL STATE FOR REWARDS
# =============================================================================
GLOBAL_AUDIO = []
GLOBAL_EXPECTED_TEXT = []
GLOBAL_REASONING = []

# =============================================================================
# REWARD FUNCTIONS
# =============================================================================
@torch.inference_mode()
def get_clap_text_embedding(text):
    inputs = clap_tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    outputs = clap_text_model(**inputs)
    mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return F.normalize(sum_emb / sum_mask, p=2, dim=1)

@torch.inference_mode()
def get_clap_audio_embedding(waveform_16k):
    target_len = 16000 * 30
    if waveform_16k.shape[0] > target_len:
        waveform_16k = waveform_16k[:target_len]
    elif waveform_16k.shape[0] < target_len:
        waveform_16k = F.pad(waveform_16k, (0, target_len - waveform_16k.shape[0]))

    inputs = clap_feature_extractor(waveform_16k.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    emb = clap_audio_model(inputs.input_features.to(DEVICE))
    return emb

INVALID_TOKEN_PENALTY = 0.1

REASONING_PATTERN = re.compile(
    rf"(.*?){re.escape(REASONING_END)}",
    re.DOTALL
)

def extract_reasoning_and_audio(content):
    """Extract reasoning text and audio tokens from model completion."""
    reasoning_match = REASONING_PATTERN.search(content)
    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""

    audio_pattern = r"<custom_token_\d+>"
    tokens = re.findall(audio_pattern, content)
    codes = tokenizer.encode("".join(tokens))[1:] if tokens else []

    return reasoning_text, codes

def reasoning_format_reward(prompts, completions, **kwargs):
    """Reward for correct format: reasoning + audio."""
    scores = []
    for completion in completions:
        content = completion[0]["content"]
        score = 0.0

        if REASONING_END in content:
            score += 1.5
        else:
            score -= 2.0

        if "<custom_token_" in content:
            score += 1.5
        else:
            score -= 1.0

        scores.append(score)
    return scores

def reasoning_length_reward(prompts, completions, **kwargs):
    """Reward good reasoning length (50-150 words optimal)."""
    scores = []
    for completion in completions:
        content = completion[0]["content"]
        reasoning_text, _ = extract_reasoning_and_audio(content)

        if not reasoning_text:
            scores.append(-1.0)
            continue

        word_count = len(reasoning_text.split())

        if word_count < 10:
            score = -1.0
        elif word_count < 30:
            score = 0.5 * (word_count / 30)
        elif word_count <= 150:
            score = 1.0
        elif word_count <= 300:
            score = 1.0 - 0.5 * ((word_count - 150) / 150)
        else:
            score = 0.0

        scores.append(score)
    return scores

def reasoning_quality_reward(prompts, completions, caption, **kwargs):
    """Reward reasoning that mentions relevant concepts."""
    caption_text = caption[0] if isinstance(caption, list) else caption
    if not caption_text:
        return [0.0] * len(completions)

    caption_lower = caption_text.lower()

    # Emotion keywords
    emotion_keywords = {
        "happy": ["happy", "joy", "cheerful", "bright", "upbeat", "smile", "excited"],
        "sad": ["sad", "melancholy", "somber", "down", "grief", "sorrow", "mournful"],
        "angry": ["angry", "fury", "rage", "intense", "forceful", "aggressive", "harsh"],
        "calm": ["calm", "peaceful", "serene", "gentle", "soft", "relaxed", "soothing"],
        "fearful": ["fear", "scared", "anxious", "nervous", "trembling", "worried"],
        "surprised": ["surprise", "shock", "astonish", "unexpected", "amazed"],
    }

    relevant_keywords = []
    for emotion, keywords in emotion_keywords.items():
        if any(kw in caption_lower for kw in keywords):
            relevant_keywords.extend(keywords)

    # Prosody terms always relevant
    prosody_terms = ["pace", "speed", "slow", "fast", "emphasis", "stress",
                    "pitch", "tone", "volume", "loud", "soft", "pause", "breath",
                    "energy", "intensity", "voice", "delivery"]

    scores = []
    for completion in completions:
        content = completion[0]["content"]
        reasoning_text, _ = extract_reasoning_and_audio(content)

        if not reasoning_text:
            scores.append(-0.5)
            continue

        reasoning_lower = reasoning_text.lower()

        matches = sum(1 for kw in relevant_keywords if kw in reasoning_lower)
        prosody_matches = sum(1 for term in prosody_terms if term in reasoning_lower)

        score = min(1.5, matches * 0.3 + prosody_matches * 0.15)
        scores.append(score)

    return scores

def wer_reward(prompts, completions, expected_text, **kwargs):
    """WER reward for speech accuracy."""
    GLOBAL_AUDIO.clear()
    GLOBAL_EXPECTED_TEXT.clear()
    GLOBAL_REASONING.clear()

    exp_text = expected_text[0] if isinstance(expected_text, list) else expected_text
    GLOBAL_EXPECTED_TEXT.append(exp_text)

    if LOCAL_RANK == 0 and len(completions) > 0:
        first_content = completions[0][0]["content"]
        print(f"[DEBUG] First completion ({len(first_content)} chars): {first_content[:500]}...")

    scores = []

    for idx, completion in enumerate(completions):
        content = completion[0]["content"]
        reasoning_text, codes = extract_reasoning_and_audio(content)

        GLOBAL_REASONING.append(reasoning_text)

        if LOCAL_RANK == 0 and idx == 0:
            print(f"[DEBUG] Reasoning ({len(reasoning_text)} chars): {reasoning_text[:200]}..." if reasoning_text else "[DEBUG] No reasoning")
            print(f"[DEBUG] Found {len(codes)} audio codes")

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

            GLOBAL_AUDIO.append(audio.cpu())

            audio_np = audio.squeeze().cpu().numpy()
            result = asr_pipe(audio_np, return_timestamps=True)
            transcribed = result["text"]

            wer_score = compute_wer(exp_text.lower(), transcribed.lower())
            accuracy = max(0, 1 - wer_score)
            final_score = accuracy - invalid_penalty
            scores.append(final_score)

        except Exception as e:
            print(f"[Rank {LOCAL_RANK}] WER error: {e}")
            scores.append(-5.0 - invalid_penalty)
            GLOBAL_AUDIO.append(None)

    return scores

def whisperclap_reward(prompts, completions, caption, **kwargs):
    """CLAP reward for audio-emotion alignment."""
    caption_text = caption[0] if isinstance(caption, list) else caption

    if not caption_text:
        return [0.0] * len(completions)

    text_emb = get_clap_text_embedding(caption_text)

    scores = []
    for audio in GLOBAL_AUDIO:
        if audio is None:
            scores.append(-1.0)
            continue

        try:
            audio_24k = audio.squeeze().to(DEVICE)
            audio_16k = RESAMPLER_24_TO_16(audio_24k)
            audio_emb = get_clap_audio_embedding(audio_16k)
            similarity = torch.matmul(audio_emb, text_emb.t()).item()
            scores.append(similarity)
        except Exception as e:
            print(f"[Rank {LOCAL_RANK}] CLAP error: {e}")
            scores.append(-1.0)

    return scores

def audiobox_reward(prompts, completions, **kwargs):
    """Audio quality reward."""
    scores = []

    valid_audios = [(i, audio) for i, audio in enumerate(GLOBAL_AUDIO) if audio is not None]

    if not valid_audios:
        GLOBAL_AUDIO.clear()
        return [-1.0] * len(completions)

    try:
        predictions = audiobox_predictor.forward([
            {"path": audio.squeeze().unsqueeze(0).to("cuda:0"), "sample_rate": 24000}
            for _, audio in valid_audios
        ])

        pred_dict = {valid_audios[i][0]: pred for i, pred in enumerate(predictions)}

        for i in range(len(completions)):
            if i in pred_dict:
                pq = pred_dict[i]["PQ"]
                score = (pq - 5) / 5
                scores.append(score)
            else:
                scores.append(-1.0)

    except Exception as e:
        print(f"[Rank {LOCAL_RANK}] Audiobox error: {e}")
        scores = [-1.0] * len(completions)

    GLOBAL_AUDIO.clear()
    return scores

# =============================================================================
# TRAINING CONFIG
# =============================================================================
training_args = GRPOConfig(
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    save_steps=100,
    report_to="wandb",
    output_dir="outputs_reasoning_grpo",
    bf16=True,
    run_name="vocalino_reasoning_grpo",

    num_generations=4,
    max_prompt_length=2048,
    max_completion_length=2048,

    temperature=0.8,
    top_p=0.95,
    top_k=50,
    min_p=0.1,

    generation_kwargs={
        "stop_token_ids": [END_OF_SPEECH, tokenizer.eos_token_id],
    },

    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.4,
    vllm_max_model_length=3000,
    vllm_tensor_parallel_size=1,
    vllm_enable_sleep_mode=True,

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

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# =============================================================================
# TRAINER
# =============================================================================
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reasoning_format_reward,      # Format correctness
        reasoning_length_reward,      # Good reasoning length
        reasoning_quality_reward,     # Mentions relevant concepts
        wer_reward,                   # Speech accuracy
        whisperclap_reward,           # Audio-emotion alignment
        audiobox_reward,              # Audio quality
    ],
    args=training_args,
    train_dataset=dataset,
)

# =============================================================================
# TRAIN
# =============================================================================
if __name__ == "__main__":
    print("Starting GRPO training...")
    print("Make sure you ran stage1_sft.py first!")
    trainer.train()

    model.save_pretrained("outputs_reasoning_grpo/final")
    tokenizer.save_pretrained("outputs_reasoning_grpo/final")
    print("Done! Model saved to outputs_reasoning_grpo/final")

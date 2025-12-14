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
MODEL_NAME = "ChristophSchuhmann/Vocalino_0.11_alpha"
DATASET_NAME = "mrfakename/emoact_prompts_with_language"
DEVICE = "cuda"
MAX_SEQ_LENGTH = 4096
SAMPLE_RATE = 24000

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

# =============================================================================
# LOAD SNAC
# =============================================================================
print("Loading SNAC...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)

# =============================================================================
# LOAD REWARD MODELS
# =============================================================================
print("Loading reward models...")

# ASR for WER
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device=DEVICE,
    torch_dtype=torch.bfloat16,
)

# Whisper-CLAP
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

# Audiobox Aesthetics
audiobox_predictor = initialize_predictor()

# Resampler for CLAP (24kHz -> 16kHz)
RESAMPLER_24_TO_16 = T.Resample(orig_freq=24000, new_freq=16000).to(DEVICE)

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

def prepare_snac_codes(token_ids, device="cuda"):
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
            print(f"SNAC decode error: {e}")
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
import os
import pickle
from tqdm import tqdm

VOICE_CACHE_PATH = "voice_prompts_cache.pkl"
MAX_REF_SECONDS = 1.5  # Shorter reference = faster training
MAX_REF_SAMPLES = int(SAMPLE_RATE * MAX_REF_SECONDS)
MAX_REF_TOKENS = 175  # ~1.5 seconds of audio
ENCODE_BATCH_SIZE = 64

# Try to load from cache first
if os.path.exists(VOICE_CACHE_PATH):
    print(f"Loading cached voice prompts from {VOICE_CACHE_PATH}...")
    with open(VOICE_CACHE_PATH, "rb") as f:
        VOICE_PROMPT_TOKENS = pickle.load(f)
    print(f"Loaded {len(VOICE_PROMPT_TOKENS)} cached voice prompts")
else:
    print("Loading voice prompts dataset...")
    voice_prompts_ds = load_dataset("mrfakename/voice_design", split="train")
    voice_prompts_ds = voice_prompts_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    print("Pre-encoding voice prompts (will be cached for next run)...")
    VOICE_PROMPT_TOKENS = []
    
    # Process one at a time to avoid batching issues
    for idx in tqdm(range(len(voice_prompts_ds)), desc="Encoding voice prompts"):
        try:
            audio_data = voice_prompts_ds[idx]["audio"]
            if not audio_data or "array" not in audio_data:
                continue
            
            wav = torch.from_numpy(audio_data["array"]).to(dtype=torch.float32)
            sr = audio_data["sampling_rate"]
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                wav = resampler(wav)
            
            # Trim to max length
            if wav.shape[0] > MAX_REF_SAMPLES:
                wav = wav[:MAX_REF_SAMPLES]
            
            # SNAC expects [batch, channels, time]
            wav = wav.unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.inference_mode():
                codes = snac_model.encode(wav)
            
            # codes is a list of 3 tensors, access like codes[layer][batch][time]
            num_frames = codes[0].shape[1]
            all_codes = []
            
            for i in range(num_frames):
                all_codes.extend([
                    codes[0][0][i].item() + TOKEN_OFFSET_BASE + LAYER_OFFSETS[0],
                    codes[1][0][2*i].item() + TOKEN_OFFSET_BASE + LAYER_OFFSETS[1],
                    codes[2][0][4*i].item() + TOKEN_OFFSET_BASE + LAYER_OFFSETS[2],
                    codes[2][0][4*i+1].item() + TOKEN_OFFSET_BASE + LAYER_OFFSETS[3],
                    codes[1][0][2*i+1].item() + TOKEN_OFFSET_BASE + LAYER_OFFSETS[4],
                    codes[2][0][4*i+2].item() + TOKEN_OFFSET_BASE + LAYER_OFFSETS[5],
                    codes[2][0][4*i+3].item() + TOKEN_OFFSET_BASE + LAYER_OFFSETS[6]
                ])
            
            if len(all_codes) > MAX_REF_TOKENS:
                all_codes = all_codes[:MAX_REF_TOKENS]
            
            if all_codes:
                VOICE_PROMPT_TOKENS.append(all_codes)
                
        except Exception as e:
            if idx < 5:  # Print first few errors for debugging
                print(f"Error encoding voice prompt {idx}: {e}")
    
    # Save to cache
    print(f"Saving {len(VOICE_PROMPT_TOKENS)} voice prompts to cache...")
    with open(VOICE_CACHE_PATH, "wb") as f:
        pickle.dump(VOICE_PROMPT_TOKENS, f)

print(f"Total voice prompts available: {len(VOICE_PROMPT_TOKENS)}")

if len(VOICE_PROMPT_TOKENS) == 0:
    raise ValueError("No voice prompts were encoded! Check the voice_design dataset.")

def process_example(example):
    """
    Create prompt with voice cloning format using random voice from voice_design dataset.
    """
    # Pick a random voice prompt
    ref_tokens = random.choice(VOICE_PROMPT_TOKENS)
    
    caption = example.get("caption", "")
    text = example.get("text", "")
    full_text = f"{caption} {text}" if caption else text
    
    # Convert ref tokens to token strings for the prompt (batch decode is faster)
    ref_token_str = tokenizer.decode(ref_tokens, skip_special_tokens=False)
    
    # Format: Reference audio: {tokens} Text: {text}
    prompt_content = f"Reference audio: {ref_token_str} Text: {text}"
    
    return {
        "prompt": [{"role": "user", "content": prompt_content}],
        "expected_text": full_text,
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
def get_clap_text_embedding(text):
    inputs = clap_tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = clap_text_model(**inputs)
        mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(sum_emb / sum_mask, p=2, dim=1)

def get_clap_audio_embedding(waveform_16k):
    target_len = 16000 * 30
    if waveform_16k.shape[0] > target_len:
        waveform_16k = waveform_16k[:target_len]
    elif waveform_16k.shape[0] < target_len:
        waveform_16k = F.pad(waveform_16k, (0, target_len - waveform_16k.shape[0]))
    
    inputs = clap_feature_extractor(waveform_16k.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        emb = clap_audio_model(inputs.input_features.to(DEVICE))
    return emb

INVALID_TOKEN_PENALTY = 0.1

def wer_reward(prompts, completions, **kwargs):
    GLOBAL_AUDIO.clear()
    GLOBAL_EXPECTED_TEXT.clear()
    
    # Extract text from prompt (after "Text: ")
    prompt_content = prompts[0][-1]["content"]
    text_match = re.search(r"Text:\s*(.+)$", prompt_content)
    expected_text = text_match.group(1) if text_match else prompt_content
    expected_text = re.sub(r"<[^>]*>", "", expected_text).strip()
    
    GLOBAL_EXPECTED_TEXT.append(expected_text)
    
    pattern = r"<custom_token_\d+>"
    scores = []
    
    for completion in completions:
        content = completion[0]["content"]
        tokens = re.findall(pattern, content)
        codes = tokenizer.encode("".join(tokens))[1:] if tokens else []
        
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
            
            GLOBAL_AUDIO.append(audio)
            
            audio_np = audio.squeeze().cpu().numpy()
            result = asr_pipe(audio_np, return_timestamps=True)
            transcribed = result["text"]
            
            wer_score = compute_wer(expected_text.lower(), transcribed.lower())
            accuracy = max(0, 1 - wer_score)
            final_score = accuracy - invalid_penalty
            scores.append(final_score)
            
        except Exception as e:
            print(f"WER error: {e}")
            scores.append(-5.0 - invalid_penalty)
            GLOBAL_AUDIO.append(None)
    
    return scores

def whisperclap_reward(prompts, completions, **kwargs):
    # Extract caption from the text (before the actual text content)
    prompt_content = prompts[0][-1]["content"]
    text_match = re.search(r"Text:\s*(.+)$", prompt_content)
    if text_match:
        full_text = text_match.group(1)
        # Caption is typically the first part before the actual speech text
        # For now, use the full text as the caption target
        caption = full_text.split(".")[0] if "." in full_text else full_text
    else:
        caption = ""
    
    if not caption:
        return [0.0] * len(completions)
    
    text_emb = get_clap_text_embedding(caption)
    
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
            print(f"CLAP error: {e}")
            scores.append(-1.0)
    
    return scores

def audiobox_reward(prompts, completions, **kwargs):
    scores = []
    
    valid_audios = [(i, audio) for i, audio in enumerate(GLOBAL_AUDIO) if audio is not None]
    
    if not valid_audios:
        GLOBAL_AUDIO.clear()
        return [-1.0] * len(completions)
    
    try:
        predictions = audiobox_predictor.forward([
            {"path": audio.squeeze().unsqueeze(0), "sample_rate": 24000}
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
        print(f"Audiobox error: {e}")
        scores = [-1.0] * len(completions)
    
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
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    save_steps=100,
    report_to="wandb",
    output_dir="outputs_laion",
    bf16=True,

    num_generations=2,  # Fewer generations = faster
    max_prompt_length=2048,
    max_completion_length=2048,

    temperature=0.8,
    top_p=0.95,
    top_k=None,
    min_p=0.1,

    generation_kwargs={
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": END_OF_SPEECH,
    },

    use_vllm=False,
    
    # FSDP Configuration
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
tokenizer.chat_template = """
{% for message in messages %}
{{ message['content'] }}
{% endfor %}
"""

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[wer_reward, whisperclap_reward, audiobox_reward],
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

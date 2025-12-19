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
LAYER_OFFSETS = [0, 4096, 8192, 8192, 4096, 8192, 8192]
AUDIO_TOKEN_END = TOKEN_OFFSET_BASE + (3 * SNAC_VOCAB_SIZE) 

# Explicit Strings for mapping
TOK_START_SPEECH = "<|startofspeech|>"
TOK_END_SPEECH = "<|endofspeech|>"
TOK_START_AI = "<|startofai|>"
TOK_START_HUMAN = "<|startofhuman|>"
TOK_END_HUMAN = "<|endofhuman|>"
TOK_END_TEXT = "<|endoftext|>"

# =============================================================================
# GLOBAL MODEL REFERENCES (initialized in main, used by reward functions)
# =============================================================================
snac_model = None
asr_pipe = None
gte_model = None
gte_tokenizer = None
majestrino_tagger = None
RESAMPLER_24_TO_16 = None

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

def decode_vocalino_audio(text_content, snac_model_ref):
    token_strings = re.findall(r"<custom_token_(\d+)>", text_content)
    
    if len(token_strings) == 0:
        return None
    
    token_ids = [int(t) for t in token_strings]
    
    # Filter valid IDs
    valid_ids = [t for t in token_ids if TOKEN_OFFSET_BASE <= t < AUDIO_TOKEN_END]
    valid_ids = valid_ids[:(len(valid_ids) // 7) * 7]
    
    if len(valid_ids) < 7: return None
    
    # De-interleave
    c0, c1, c2 = [], [], []
    for i in range(len(valid_ids) // 7):
        chunk = valid_ids[i*7 : (i+1)*7]
        idx0 = chunk[0] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[0]
        idx1_a = chunk[1] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[1]
        idx1_b = chunk[4] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[4]
        idx2_a = chunk[2] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[2]
        idx2_b = chunk[3] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[3]
        idx2_c = chunk[5] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[5]
        idx2_d = chunk[6] - TOKEN_OFFSET_BASE - LAYER_OFFSETS[6]
        
        c0.append(idx0)
        c1.extend([idx1_a, idx1_b])
        c2.extend([idx2_a, idx2_b, idx2_c, idx2_d])

    # Sanitize
    c0 = sanitize_indices(c0)
    c1 = sanitize_indices(c1)
    c2 = sanitize_indices(c2)

    try:
        with torch.no_grad():
            z0 = torch.tensor(c0, device=SNAC_DEVICE).unsqueeze(0)
            z1 = torch.tensor(c1, device=SNAC_DEVICE).unsqueeze(0)
            z2 = torch.tensor(c2, device=SNAC_DEVICE).unsqueeze(0)
            audio = snac_model_ref.decode([z0, z1, z2])
            return audio.squeeze().numpy()
    except Exception as e:
        return None

# =============================================================================
# REWARDS
# =============================================================================

def combined_reward(prompts, completions, expected_text, caption, **kwargs):
    """Combined reward function that computes both WER and CLAP rewards in a single pass."""
    global snac_model, asr_pipe, gte_model, gte_tokenizer, majestrino_tagger, RESAMPLER_24_TO_16
    
    scores = []
    
    if LOCAL_RANK == 0:
        log_debug(f"Target Text (first): '{expected_text[0][:30] if expected_text else 'N/A'}...'")

    for i, completion in enumerate(completions):
        if isinstance(completion, list): content = completion[0]["content"]
        else: content = str(completion)

        audio_np = decode_vocalino_audio(content, snac_model)
        
        if audio_np is None:
            # Massive penalty for text hallucination
            if len(content) > 200:
                scores.append(-2.0) 
            else:
                scores.append(-1.0)
            continue
        
        # =====================================================================
        # WER REWARD COMPONENT
        # =====================================================================
        wer_score = -1.0
        try:
            # Resample from 24kHz to 16kHz for Whisper
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(DEVICE)
            audio_16k = RESAMPLER_24_TO_16(audio_tensor).cpu().numpy()
            
            transcribed = asr_pipe(audio_16k, generate_kwargs={"language": "en"})["text"].lower().strip()
            target = expected_text[i].lower().strip() if isinstance(expected_text, list) else expected_text.lower().strip()
            wer_val = compute_wer(target, transcribed)
            wer_score = max(-1.0, 1.0 - wer_val)
            
            if LOCAL_RANK == 0 and i == 0:
                log_debug(f"WER: {wer_val:.2f} | WER Score: {wer_score:.2f} | Tx: '{transcribed[:30]}...'")
                
        except Exception as e:
            if LOCAL_RANK == 0:
                log_debug(f"WER error: {e}")
            wer_score = -1.0
        
        # =====================================================================
        # CLAP REWARD COMPONENT
        # =====================================================================
        clap_score = -1.0
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
                clap_score = -0.5
            else:
                # Get expected caption
                expected_prompt = caption[i] if isinstance(caption, list) else caption
                
                # Get embeddings for expected prompt
                expected_inputs = gte_tokenizer(expected_prompt, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    expected_emb = F.normalize(gte_model(**expected_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)
                
                # Get embeddings for tags
                tags_inputs = gte_tokenizer(tags, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    tags_emb = F.normalize(gte_model(**tags_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)
                
                # Compute similarity (ranges from -1 to 1)
                clap_score = torch.matmul(expected_emb, tags_emb.t()).item()
                
                if LOCAL_RANK == 0 and i == 0:
                    log_debug(f"CLAP Score: {clap_score:.2f} | Tags: '{tags[:50]}...'")
                    
        except Exception as e:
            if LOCAL_RANK == 0:
                log_debug(f"CLAP error: {e}")
            clap_score = -1.0
        
        # =====================================================================
        # COMBINE SCORES
        # =====================================================================
        # Weight WER more heavily since it's the primary objective
        combined_score = 0.7 * wer_score + 0.3 * clap_score
        
        if LOCAL_RANK == 0 and i == 0:
            log_debug(f"Combined Score: {combined_score:.2f} (WER: {wer_score:.2f}, CLAP: {clap_score:.2f})")
        
        scores.append(combined_score)
    
    return scores

# =============================================================================
# TOKENIZER PATCHING (CRITICAL FIX)
# =============================================================================
def patch_tokenizer(tokenizer):
    """
    Force-register special tokens with specific strings so we can construct
    prompts that survive the ID->String->ID roundtrip.
    """
    # Define the tokens
    special_tokens = [
        TOK_START_SPEECH, TOK_END_SPEECH, 
        TOK_START_AI, TOK_START_HUMAN, TOK_END_HUMAN
    ]
    
    # Add them (this usually appends them to the end of vocab)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    # We don't necessarily need them to map to 128257 if we re-train embeddings,
    # but for GRPO we just need them to BE tokens.
    # The key is consistent usage.
    return tokenizer

def format_vocalino_prompt(example):
    caption = example['caption']
    text = example['text']
    
    # Construct prompt using the EXPLICIT STRINGS we registered
    # This guarantees the tokenizer won't split them into garbage text
    prompt_str = (
        f"{TOK_START_HUMAN}"
        f"<start_of_caption>{caption}<end_of_caption>{text}"
        f"{TOK_END_HUMAN}"
        f"{TOK_START_AI}{TOK_START_SPEECH}" # <--- The Trigger
    )
    
    return {
        "prompt": prompt_str,
        "expected_text": text,
        "caption": caption,
    }

# =============================================================================
# HELPER MODEL INITIALIZATION
# =============================================================================
def init_helper_models():
    """Initialize all helper models. Called once per process."""
    global snac_model, asr_pipe, gte_model, gte_tokenizer, majestrino_tagger, RESAMPLER_24_TO_16
    
    log_debug(f"Loading SNAC on {SNAC_DEVICE}...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(SNAC_DEVICE)
    
    log_debug("Loading ASR pipeline...")
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-base", 
        device=LOCAL_RANK, 
        torch_dtype=torch.bfloat16,
        # Whisper expects 16kHz audio - we'll resample before passing
    )
    
    log_debug("Loading GTE model for similarity...")
    gte_model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(DEVICE).eval()
    gte_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
    
    log_debug("Loading Majestrino tagger...")
    majestrino_tagger = MajestrinoTagger.from_pretrained()
    majestrino_tagger.load_tags()
    
    log_debug("Loading resampler...")
    RESAMPLER_24_TO_16 = T.Resample(orig_freq=24000, new_freq=16000).to(DEVICE)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Initialize helper models (sets global variables)
    init_helper_models()
    
    # 1. Load Tokenizer & Patch
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    patch_tokenizer(tokenizer)
    
    # Fix padding
    if tokenizer.eos_token_id is None: tokenizer.eos_token_id = 128009
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 128009
    tokenizer.padding_side = "left"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]

    # 2. Load Model & Resize Embeddings for new tokens
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.bfloat16, use_cache=False)
    model.resize_token_embeddings(len(tokenizer)) # Necessary because we added tokens

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
        vllm_gpu_memory_utilization=0.7,
        report_to="wandb",
        remove_unused_columns=False,
        # FSDP config for multi-GPU training
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": False,
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        },
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(format_vocalino_prompt)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[combined_reward],  # Single combined reward function
        args=training_args,
        train_dataset=dataset,
    )

    log_debug("Starting Training...")
    trainer.train()
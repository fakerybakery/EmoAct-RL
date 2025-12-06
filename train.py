import re
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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from trl import GRPOConfig, GRPOTrainer
from jiwer import wer as compute_wer
from audiobox_aesthetics.infer import initialize_predictor

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "canopylabs/orpheus-3b-0.1-ft"
DATASET_NAME = "mrfakename/emoact_prompts_with_language"  # HF dataset with: text, caption, audio
DEVICE = "cuda"
MAX_SEQ_LENGTH = 4096
SAMPLE_RATE = 24000

# =============================================================================
# SPECIAL TOKENS
# =============================================================================
TOKENIZER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009

START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7

# New tokens for caption
START_OF_CAPTION = TOKENIZER_LENGTH + 8
END_OF_CAPTION = TOKENIZER_LENGTH + 9

AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10

# =============================================================================
# LOAD BASE MODEL (FFT)
# =============================================================================
print("Loading model for full fine-tuning...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add special tokens for caption
special_tokens = {
    "additional_special_tokens": [
        "<start_of_caption>",
        "<end_of_caption>",
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

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
# AUDIO DECODING (for reward computation)
# =============================================================================
def prepare_snac_codes(token_ids, device="cuda"):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    if len(token_ids) < 7:
        empty = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)
        return [empty.clone(), empty.clone(), empty.clone()]
    
    new_length = (len(token_ids) // 7) * 7
    trimmed = token_ids[:new_length]
    processed = [t - 128266 for t in trimmed]
    
    layer_1, layer_2, layer_3 = [], [], []
    
    for i in range(len(processed) // 7):
        base = 7 * i
        layer_1.append(processed[base])
        layer_2.append(processed[base + 1] - 4096)
        layer_3.append(processed[base + 2] - 2 * 4096)
        layer_3.append(processed[base + 3] - 3 * 4096)
        layer_2.append(processed[base + 4] - 4 * 4096)
        layer_3.append(processed[base + 5] - 5 * 4096)
        layer_3.append(processed[base + 6] - 6 * 4096)
    
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
    
    segments = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        codes = prepare_snac_codes(chunk, device=DEVICE)
        
        if codes[0].numel() == 0:
            continue
        
        with torch.no_grad():
            audio = snac_model.decode(codes)
            if audio.ndim == 2:
                audio = audio.unsqueeze(1)
            segments.append(audio.cpu())
    
    if not segments:
        return None
    
    return torch.cat(segments, dim=2)

# =============================================================================
# DATASET PREPARATION
# =============================================================================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# Create prompt format (no audio needed - model generates it, rewards score it)
dataset = dataset.map(
    lambda x: {
        "prompt": [{"role": "user", "content": f"{x['voice']}: <start_of_caption>{x['caption']}<end_of_caption>{x['text']}"}],
    }
)

# =============================================================================
# GLOBAL STATE FOR REWARDS
# =============================================================================
GLOBAL_AUDIO = []

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
    # waveform should be 16kHz, shape [samples]
    target_len = 16000 * 30
    if waveform_16k.shape[0] > target_len:
        waveform_16k = waveform_16k[:target_len]
    elif waveform_16k.shape[0] < target_len:
        waveform_16k = F.pad(waveform_16k, (0, target_len - waveform_16k.shape[0]))
    
    inputs = clap_feature_extractor(waveform_16k.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        emb = clap_audio_model(inputs.input_features.to(DEVICE))
    return emb

def wer_reward(prompts, completions, **kwargs):
    GLOBAL_AUDIO.clear()
    
    # Extract text from prompt (after caption)
    prompt_content = prompts[0][-1]["content"]
    # Remove caption tags to get just the text
    text_match = re.search(r"<end_of_caption>(.+)$", prompt_content)
    expected_text = text_match.group(1) if text_match else prompt_content
    expected_text = re.sub(r"<[^>]*>", "", expected_text)  # Remove any remaining tags
    
    pattern = r"<custom_token_\d+>"
    scores = []
    
    for completion in completions:
        content = completion[0]["content"]
        tokens = re.findall(pattern, content)
        codes = tokenizer.encode("".join(tokens).replace("<custom_token_4><custom_token_5><custom_token_1>", "").replace("<custom_token_2><custom_token_6><custom_token_3>", ""))[1:]
        
        if not codes:
            scores.append(-1.0)
            GLOBAL_AUDIO.append(None)
            continue
        
        try:
            audio = decode_tokens_to_audio(codes)
            if audio is None:
                scores.append(-1.0)
                GLOBAL_AUDIO.append(None)
                continue
            
            GLOBAL_AUDIO.append(audio)
            
            # ASR
            audio_np = audio.squeeze().cpu().numpy()
            result = asr_pipe(audio_np)
            transcribed = result["text"]
            
            # WER -> accuracy
            wer_score = compute_wer(expected_text.lower(), transcribed.lower())
            accuracy = max(0, 1 - wer_score)
            scores.append(accuracy)
            
        except Exception as e:
            print(f"WER error: {e}")
            scores.append(-1.0)
            GLOBAL_AUDIO.append(None)
    
    return scores

def whisperclap_reward(prompts, completions, **kwargs):
    # Extract caption from prompt
    prompt_content = prompts[0][-1]["content"]
    caption_match = re.search(r"<start_of_caption>(.+?)<end_of_caption>", prompt_content)
    caption = caption_match.group(1) if caption_match else ""
    
    if not caption:
        return [0.0] * len(completions)
    
    # Get text embedding for caption
    text_emb = get_clap_text_embedding(caption)
    
    scores = []
    for audio in GLOBAL_AUDIO:
        if audio is None:
            scores.append(-1.0)
            continue
        
        try:
            # Resample to 16kHz for CLAP
            audio_24k = audio.squeeze().to(DEVICE)
            audio_16k = RESAMPLER_24_TO_16(audio_24k)
            
            # Get audio embedding
            audio_emb = get_clap_audio_embedding(audio_16k)
            
            # Cosine similarity
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
                # Normalize: PQ is 0-10, we want reward centered around 0
                score = (pq - 5) / 5  # Maps 0-10 to -1 to 1
                scores.append(score)
            else:
                scores.append(-1.0)
                
    except Exception as e:
        print(f"Audiobox error: {e}")
        scores = [-1.0] * len(completions)
    
    GLOBAL_AUDIO.clear()
    return scores

# =============================================================================
# TRAINING CONFIG
# =============================================================================
training_args = GRPOConfig(
    # ---- training hyperparams ----
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
    output_dir="outputs",
    bf16=True,

    # ---- GRPO generation settings ----
    num_generations=4,
    max_prompt_length=2048,
    max_completion_length=2048,
    temperature=0.8,
    top_p=0.95,
    top_k=None,      # None = no top-k, equivalent to your previous -1
    min_p=0.1,

    # extra SamplingParams args go here when using vLLM
    generation_kwargs={
        "stop": [tokenizer.eos_token],
        "include_stop_str_in_output": True,
    },

    # ---- turn on vLLM in "colocate" mode ----
    use_vllm=True,
    vllm_mode="colocate",          # avoids needing an external vLLM server
    # vllm_gpu_memory_utilization=0.3,  # optional; 0.3 is the default in main
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
    
    model.save_pretrained("outputs/final")
    tokenizer.save_pretrained("outputs/final")


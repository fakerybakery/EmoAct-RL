import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor, AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "laion/whisper-clap-version-0.1"
WHISPER_BASE = "openai/whisper-small"
TEXT_BASE = "Alibaba-NLP/gte-base-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Audio model
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

# Load models
audio_model = WhisperClapModel(WHISPER_BASE).to(DEVICE)
model_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
state_dict = load_file(model_path)
clean_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
audio_model.load_state_dict(clean_state, strict=False)
audio_model.eval()

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_BASE)
tokenizer = AutoTokenizer.from_pretrained(TEXT_BASE)
text_model = AutoModel.from_pretrained(TEXT_BASE, trust_remote_code=True).to(DEVICE).eval()

# === INPUT ===
audio_path = "happy.wav"
caption = "A person speaking with a happy voice"

# Get audio embedding
wav, sr = torchaudio.load(audio_path)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
if wav.shape[0] > 1:
    wav = torch.mean(wav, dim=0, keepdim=True)
target_len = 16000 * 30
if wav.shape[1] > target_len:
    wav = wav[:, :target_len]
elif wav.shape[1] < target_len:
    wav = F.pad(wav, (0, target_len - wav.shape[1]))

inputs = feature_extractor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    audio_emb = audio_model(inputs.input_features.to(DEVICE))

# Get text embedding
text_inputs = tokenizer(caption, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = text_model(**text_inputs)
    mask = text_inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    text_emb = F.normalize(sum_emb / sum_mask, p=2, dim=1)

# Cosine similarity
score = torch.matmul(audio_emb, text_emb.t()).item()

print(f"Audio: {audio_path}")
print(f"Caption: {caption}")
print(f"Score: {score:.4f}")

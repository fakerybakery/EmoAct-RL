"""
Generate voice prompts cache for train_laion.py.
Run this ONCE on GPU 0 before starting distributed training.

Usage:
    python generate_voice_cache.py
"""

import os
import pickle
import torch
import torchaudio.transforms as T
from datasets import Audio, load_dataset
from snac import SNAC
from tqdm import tqdm

# Config
DEVICE = "cuda:0"
SAMPLE_RATE = 24000
MAX_REF_SECONDS = 1.5
MAX_REF_SAMPLES = int(SAMPLE_RATE * MAX_REF_SECONDS)
MAX_REF_TOKENS = 175  # ~1.5 seconds of audio
VOICE_CACHE_PATH = "voice_prompts_cache.pkl"

# Token offsets (must match train_laion.py)
TOKEN_OFFSET_BASE = 128266
LAYER_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]

def main():
    if os.path.exists(VOICE_CACHE_PATH):
        print(f"Cache already exists at {VOICE_CACHE_PATH}")
        with open(VOICE_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        print(f"Contains {len(cached)} voice prompts")
        return

    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)

    print("Loading voice prompts dataset...")
    voice_prompts_ds = load_dataset("mrfakename/voice_design", split="train")
    voice_prompts_ds = voice_prompts_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print("Encoding voice prompts...")
    voice_prompt_tokens = []

    for idx in tqdm(range(len(voice_prompts_ds)), desc="Encoding"):
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
                voice_prompt_tokens.append(all_codes)

        except Exception as e:
            if idx < 5:
                print(f"Error encoding voice prompt {idx}: {e}")

    print(f"Saving {len(voice_prompt_tokens)} voice prompts to {VOICE_CACHE_PATH}...")
    with open(VOICE_CACHE_PATH, "wb") as f:
        pickle.dump(voice_prompt_tokens, f)

    print("Done!")

if __name__ == "__main__":
    main()

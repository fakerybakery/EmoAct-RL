import torch
import torchaudio
from audiobox_aesthetics import infer as aes_infer

audio_path = "sad.wav"

aes_predictor = aes_infer.initialize_predictor()

audio_samples, sr = torchaudio.load(audio_path)

scores = aes_predictor.forward([{"path": audio_samples, "sample_rate": sr}])[0]

print(f"Audio: {audio_path}")
print(f"Production Quality: {scores['PQ']:.2f}")
print(f"Production Complexity: {scores['PC']:.2f}")
print(f"Content Enjoyment: {scores['CE']:.2f}")
print(f"Content Usefulness: {scores['CU']:.2f}")


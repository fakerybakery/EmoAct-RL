import torch
from transformers import pipeline
from jiwer import wer

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipe = pipeline('automatic-speech-recognition', model='openai/whisper-tiny', device=device)

audio_path = "sad.wav"
expected_text = "Sometimes I still walk past the places we used to sit and talk. I can almost hear your laugh echoing in the back of my mind. Is it stupid? That a part of me is still waiting for you to come back?"

result = pipe(audio_path)
transcribed_text = result["text"]

wer_score = wer(expected_text.lower(), transcribed_text.lower())
accuracy = max(0, 1 - wer_score)

print(f"Audio: {audio_path}")
print(f"Expected: {expected_text}")
print(f"Transcribed: {transcribed_text}")
print(f"Accuracy: {accuracy:.4f}")


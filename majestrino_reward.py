import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from majestrino_tagger import MajestrinoTagger

# Load GTE model for similarity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gte_model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(DEVICE).eval()
gte_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")

# Load pretrained model and bundled tags
tagger = MajestrinoTagger.from_pretrained()
tagger.load_tags()  # Uses bundled tags

# Tag an audio file
results = tagger.tag("samples/sad.wav", threshold=95.0, top_n_per_category=3)

categories = [r['category'] for r in results]

# expected_prompt = 'The audio is a sad song about a person who is feeling sad.'
expected_prompt = 'Happy person'
print(expected_prompt)
tags = ', '.join([r['label'] for r in results if not r['category'] in ['Content Rating']])

# Get embeddings for expected prompt
expected_inputs = gte_tokenizer(expected_prompt, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    expected_emb = F.normalize(gte_model(**expected_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)

# Get embeddings for tags
tags_inputs = gte_tokenizer(tags, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    tags_emb = F.normalize(gte_model(**tags_inputs).last_hidden_state.mean(dim=1), p=2, dim=1)

# Compute similarity
similarity = torch.matmul(expected_emb, tags_emb.t()).item()

print(categories)
print(tags)
print(f"Similarity: {similarity:.4f}")
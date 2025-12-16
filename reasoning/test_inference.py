"""
Quick test to see if the model learned the reasoning format after SFT.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "outputs_reasoning_sft/final"
DEVICE = "cuda"
SAMPLE_RATE = 24000

# SNAC constants
SNAC_VOCAB_SIZE = 4096
AUDIO_TOKEN_START = 128266

# Reasoning markers
REASONING_START = "<start_of_reasoning>"
REASONING_END = "<end_of_reasoning>"

# =============================================================================
# LOAD MODELS
# =============================================================================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

print("Loading SNAC...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE).eval()

# =============================================================================
# SNAC DECODING
# =============================================================================
def is_valid_audio_token(token_id):
    return AUDIO_TOKEN_START <= token_id < AUDIO_TOKEN_START + (7 * SNAC_VOCAB_SIZE)

def decode_tokens_to_audio(token_ids):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # Filter valid audio tokens
    token_ids = [t for t in token_ids if is_valid_audio_token(t)]
    token_ids = token_ids[:(len(token_ids) // 7) * 7]

    if len(token_ids) < 7:
        return None

    layer_1, layer_2, layer_3 = [], [], []

    for i in range(len(token_ids) // 7):
        base = 7 * i

        l1 = token_ids[base] - AUDIO_TOKEN_START
        l2_a = token_ids[base + 1] - AUDIO_TOKEN_START - SNAC_VOCAB_SIZE
        l3_a = token_ids[base + 2] - AUDIO_TOKEN_START - (2 * SNAC_VOCAB_SIZE)
        l3_b = token_ids[base + 3] - AUDIO_TOKEN_START - (3 * SNAC_VOCAB_SIZE)
        l2_b = token_ids[base + 4] - AUDIO_TOKEN_START - (4 * SNAC_VOCAB_SIZE)
        l3_c = token_ids[base + 5] - AUDIO_TOKEN_START - (5 * SNAC_VOCAB_SIZE)
        l3_d = token_ids[base + 6] - AUDIO_TOKEN_START - (6 * SNAC_VOCAB_SIZE)

        all_vals = [l1, l2_a, l2_b, l3_a, l3_b, l3_c, l3_d]
        if not all(0 <= v < SNAC_VOCAB_SIZE for v in all_vals):
            continue

        layer_1.append(l1)
        layer_2.extend([l2_a, l2_b])
        layer_3.extend([l3_a, l3_b, l3_c, l3_d])

    if not layer_1:
        return None

    codes = [
        torch.tensor(layer_1, dtype=torch.long, device=DEVICE).unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.long, device=DEVICE).unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.long, device=DEVICE).unsqueeze(0),
    ]

    with torch.no_grad():
        audio = snac_model.decode(codes)

    return audio.squeeze().cpu()

# =============================================================================
# TEST EXAMPLES
# =============================================================================
test_examples = [
    {
        "caption": "Angry man shouting",
        "text": "Get out of my house right now!",
        "voice": "tara",
    },
    {
        "caption": "Sad young woman, hesitant and quiet",
        "text": "I don't think I can do this anymore...",
        "voice": "tara",
    },
    {
        "caption": "Excited child on Christmas morning",
        "text": "Oh my god, is that a puppy?!",
        "voice": "tara",
    },
    {
        "caption": "Cookie Monster asking for cookies",
        "text": "Me want cookie! Cookie! Om nom nom!",
        "voice": "tara",
    },
]

# =============================================================================
# INFERENCE
# =============================================================================
print("\n" + "="*60)
print("TESTING REASONING FORMAT")
print("="*60)

for i, example in enumerate(test_examples):
    caption = example["caption"]
    text = example["text"]
    voice = example["voice"]

    # Build prompt in the same format as training (WITH reasoning markers, but empty reasoning)
    # The model should learn to fill in reasoning
    prompt = f"{voice}: <start_of_caption>{caption}<end_of_caption>{REASONING_START}"

    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated_ids = output[0][input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Check if it has the right format
    has_reasoning_end = REASONING_END in generated_text
    has_audio_tokens = any(is_valid_audio_token(t) for t in generated_ids)

    # Count audio tokens
    audio_token_count = sum(1 for t in generated_ids if is_valid_audio_token(t))

    print(f"\n--- Example {i+1} ---")
    print(f"Caption: {caption}")
    print(f"Text: {text}")
    print(f"\nGenerated:")
    # Show first 800 chars to see reasoning
    display_text = generated_text[:800] + ("..." if len(generated_text) > 800 else "")
    print(display_text)
    print(f"\n✓ Has {REASONING_END}: {has_reasoning_end}")
    print(f"✓ Has audio tokens: {has_audio_tokens} ({audio_token_count} tokens)")

    # Try to decode and save audio for first example
    if i == 0 and has_audio_tokens:
        audio = decode_tokens_to_audio(generated_ids)
        if audio is not None:
            sf.write("test_output.wav", audio.numpy(), SAMPLE_RATE)
            print(f"✓ Saved audio to test_output.wav ({audio.shape[-1] / SAMPLE_RATE:.2f}s)")

print("\n" + "="*60)
print("DONE")
print("="*60)

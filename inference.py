import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

# =============================================================================
# CONFIG
# =============================================================================
CHECKPOINT_PATH = "outputs/checkpoint-1000"  # Path to your saved checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000

# SNAC constants
SNAC_VOCAB_SIZE = 4096
AUDIO_TOKEN_START = 128266

# =============================================================================
# LOAD MODELS
# =============================================================================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
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
# GENERATION
# =============================================================================
def generate_speech(text, voice="tara", caption="", max_new_tokens=2048):
    # Format prompt
    if caption:
        prompt = f"{voice}: <start_of_caption>{caption}<end_of_caption>{text}"
    else:
        prompt = f"{voice}: {text}"
    
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only the generated tokens
    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
    
    # Decode to audio
    audio = decode_tokens_to_audio(generated_ids)
    
    return audio

def save_audio(audio, path, sample_rate=SAMPLE_RATE):
    if audio is None:
        print("No audio to save")
        return
    
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    sf.write(path, audio.numpy().T, sample_rate)
    print(f"Saved audio to {path}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Example usage
    text = "Hello, how are you today? I hope you're having a wonderful day."
    voice = "tara"
    caption = "A cheerful and friendly voice"
    
    print(f"Generating speech for: {text}")
    print(f"Voice: {voice}")
    print(f"Caption: {caption}")
    
    audio = generate_speech(text, voice=voice, caption=caption)
    
    if audio is not None:
        save_audio(audio, "output.wav")
        print(f"Audio duration: {audio.shape[-1] / SAMPLE_RATE:.2f}s")
    else:
        print("Failed to generate audio")


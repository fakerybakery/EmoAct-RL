import torch
import torchaudio.transforms as T
from datasets import Audio, load_dataset
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "canopylabs/orpheus-3b-0.1-ft"
DATASET_NAME = "mrfakename/voice-acting"
DEVICE = "cuda"
SAMPLE_RATE = 24000
MAX_SEQ_LENGTH = 4096

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

# =============================================================================
# LOAD MODEL
# =============================================================================
print("Loading model...")
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
# AUDIO TOKENIZATION
# =============================================================================
def tokenize_audio(waveform, orig_sr):
    waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
    
    # Resample to 24kHz if needed
    if orig_sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    waveform = waveform.unsqueeze(0).to(DEVICE)
    
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))
    
    return all_codes

def remove_duplicate_frames(codes_list):
    if len(codes_list) % 7 != 0:
        return codes_list
    
    result = codes_list[:7]
    
    for i in range(7, len(codes_list), 7):
        if codes_list[i] != result[-7]:
            result.extend(codes_list[i:i + 7])
    
    return result

# =============================================================================
# DATASET
# =============================================================================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

def process_example(example):
    try:
        audio_data = example.get("audio")
        if not audio_data or "array" not in audio_data:
            return {"input_ids": None, "labels": None, "attention_mask": None}
        
        # Tokenize audio
        codes_list = tokenize_audio(audio_data["array"], audio_data["sampling_rate"])
        codes_list = remove_duplicate_frames(codes_list)
        
        if not codes_list:
            return {"input_ids": None, "labels": None, "attention_mask": None}
        
        # Format: voice: <caption>caption</caption> text
        voice = example.get("voice", "")
        caption = example.get("caption", "")
        text = example.get("text", "")
        
        if voice:
            text_prompt = f"{voice}: <start_of_caption>{caption}<end_of_caption>{text}"
        else:
            text_prompt = f"<start_of_caption>{caption}<end_of_caption>{text}"
        
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        
        input_ids = (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
            + codes_list
            + [END_OF_SPEECH]
            + [END_OF_AI]
        )
        
        # Truncate if too long
        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
        
        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        
    except Exception as e:
        print(f"Error processing example: {e}")
        return {"input_ids": None, "labels": None, "attention_mask": None}

print("Processing dataset...")
dataset = dataset.map(process_example, remove_columns=dataset.column_names)

# Filter out failed examples
dataset = dataset.filter(lambda x: x["input_ids"] is not None)

print(f"Dataset size: {len(dataset)}")

# =============================================================================
# TRAINING
# =============================================================================
training_args = TrainingArguments(
    output_dir="outputs_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    report_to="wandb",
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# =============================================================================
# TRAIN
# =============================================================================
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained("outputs_sft/final")
    tokenizer.save_pretrained("outputs_sft/final")
    print("Done!")


"""
Generate reasoning data for SFT using Kimi-K2 via HuggingFace Inference API.
Uses joblib for concurrent generation with tqdm progress bar.
"""

import os
import json
import random
from datasets import load_dataset, Dataset
from huggingface_hub import InferenceClient, get_token
from joblib import Parallel, delayed
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
SOURCE_DATASET = "mrfakename/voice-acting"  # Has caption, text, audio
OUTPUT_PATH = "reasoning_sft_data.json"
NUM_SAMPLES = 500
N_JOBS = 32  # Concurrent requests

client = InferenceClient(
    api_key=get_token(),
    bill_to="TTS-AGI",
)

SYSTEM_PROMPT = """You are an expert voice director and speech coach. Given a caption describing the emotional tone/style and the text to be spoken, generate a brief reasoning block that explains HOW to voice this text.

Your reasoning should cover:
- Vocal qualities (pitch, tone, breathiness, roughness)
- Pacing and rhythm (fast, slow, pauses, emphasis)
- Emotional delivery (energy level, intensity)
- Specific words or phrases to emphasize
- Any character-specific traits if applicable

Keep it concise (50-150 words). Be specific and actionable, not vague.

Output ONLY the reasoning, no preamble or explanation."""

def generate_reasoning(caption: str, text: str) -> str | None:
    """Generate reasoning for a single example."""
    try:
        prompt = f"Caption: {caption}\nText: {text}"

        completion = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905:groq",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )

        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return None

def process_example(example: dict) -> dict | None:
    """Process a single example, generating reasoning."""
    caption = example.get("caption", "")
    text = example.get("text", "")

    if not caption or not text:
        return None

    reasoning = generate_reasoning(caption, text)

    if not reasoning:
        return None

    return {
        "caption": caption,
        "text": text,
        "reasoning": reasoning,
        # Keep audio reference for later pairing
        "voice": example.get("voice", ""),
    }

def main():
    print("Loading dataset...")
    dataset = load_dataset(SOURCE_DATASET, split="train")

    # Sample more than we need in case some fail
    num_to_sample = min(len(dataset), NUM_SAMPLES + 100)
    indices = random.sample(range(len(dataset)), num_to_sample)
    samples = [dataset[i] for i in indices]

    print(f"Generating reasoning for {len(samples)} examples with {N_JOBS} workers...")

    results = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(process_example)(sample)
        for sample in tqdm(samples, desc="Generating reasoning")
    )

    # Filter out failures and limit to NUM_SAMPLES
    valid_results = [r for r in results if r is not None][:NUM_SAMPLES]

    print(f"Successfully generated {len(valid_results)} examples")

    # Save to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(valid_results, f, indent=2)

    print(f"Saved to {OUTPUT_PATH}")

    # Also save as HF dataset for easy loading
    hf_dataset = Dataset.from_list(valid_results)
    hf_dataset.save_to_disk("reasoning_sft_dataset")
    print("Saved HF dataset to reasoning_sft_dataset/")

    # Print a few examples
    print("\n" + "="*50)
    print("Example outputs:")
    print("="*50)
    for i, example in enumerate(valid_results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Caption: {example['caption']}")
        print(f"Text: {example['text']}")
        print(f"Reasoning: {example['reasoning']}")

if __name__ == "__main__":
    main()

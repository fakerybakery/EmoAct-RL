import json
import itertools

# =========================
# Speech dimensions
# =========================

ages = [
    "child (approximately 4–11 years old)",
    "teen (approximately 12–17 years old)",
    "young adult (approximately 18–30 years old)",
    "middle-aged adult (approximately 31–55 years old)",
    "elderly adult (approximately 56–75 years old)",
    "very old or frail elderly adult (75+ years old)"
]

softness_levels = [
    "very soft and gentle vocal effort",
    "soft vocal effort",
    "neutral vocal effort",
    "firm vocal effort",
    "harsh or pressed vocal effort"
]

resonance_placements = [
    "chest-dominant resonance",
    "chest-balanced resonance",
    "balanced resonance",
    "head-balanced resonance",
    "head-dominant resonance"
]

nasality_levels = [
    "no nasality (fully oral resonance)",
    "slight nasality",
    "moderate nasality",
    "strong nasality"
]

phonation_types = [
    "modal phonation",
    "breathy phonation",
    "slightly creaky phonation",
    "pressed phonation"
]

speech_rates = [
    "very slow speech rate",
    "slow speech rate",
    "moderate speech rate",
    "fast speech rate",
    "very fast speech rate"
]

# =========================
# EmoNet emotion taxonomy
# (categories + full word lists)
# =========================

emotion_taxonomy = {
    "Amusement": [
        "amusement", "mirth", "joviality", "playfulness",
        "silliness", "lighthearted fun", "jesting", "laughter"
    ],
    "Elation": [
        "happiness", "joy", "excitement", "exhilaration",
        "delight", "jubilation", "bliss", "cheerfulness"
    ],
    "Pleasure / Ecstasy": [
        "pleasure", "ecstasy", "rapture", "beatitude", "intense bliss"
    ],
    "Contentment": [
        "contentment", "calmness", "peacefulness", "relaxation",
        "satisfaction", "ease", "serenity", "tranquility", "fulfillment"
    ],
    "Thankfulness / Gratitude": [
        "gratitude", "thankfulness", "appreciation", "gratefulness"
    ],
    "Affection": [
        "affection", "warmth", "compassion", "tenderness",
        "caring", "sympathy", "trust", "devotion"
    ],
    "Infatuation": [
        "infatuation", "romantic desire", "having a crush",
        "adoration", "fondness", "butterflies in the stomach"
    ],
    "Hope / Optimism": [
        "hope", "optimism", "enthusiasm", "anticipation",
        "encouragement", "inspiration", "determination"
    ],
    "Triumph": [
        "triumph", "victory", "superiority", "conquest"
    ],
    "Pride": [
        "pride", "self-confidence", "dignity", "honor"
    ],
    "Interest": [
        "interest", "curiosity", "fascination", "intrigue"
    ],
    "Awe": [
        "awe", "wonder", "amazement", "reverence"
    ],
    "Astonishment / Surprise": [
        "astonishment", "surprise", "shock", "amazement"
    ],
    "Relief": [
        "relief", "comfort", "reassurance"
    ],
    "Sadness / Grief": [
        "sadness", "grief", "sorrow", "heartache", "despair"
    ],
    "Disappointment": [
        "disappointment", "letdown", "dissatisfaction"
    ],
    "Distress": [
        "distress", "anguish", "emotional pain", "upset"
    ],
    "Fear": [
        "fear", "anxiety", "dread", "terror", "fright"
    ],
    "Helplessness": [
        "helplessness", "powerlessness", "vulnerability"
    ],
    "Bitterness": [
        "bitterness", "resentment", "sour feelings"
    ],
    "Contempt": [
        "contempt", "disdain", "scorn"
    ],
    "Disgust": [
        "disgust", "revulsion", "aversion"
    ],
    "Shame": [
        "shame", "humiliation", "self-conscious embarrassment"
    ],
    "Doubt": [
        "doubt", "uncertainty", "indecision"
    ],
    "Confusion": [
        "confusion", "bewilderment", "perplexity"
    ],
    "Longing": [
        "longing", "yearning", "wistfulness"
    ],
    "Jealousy / Envy": [
        "jealousy", "envy", "begrudging desire"
    ],
    "Pain": [
        "pain", "hurt", "discomfort"
    ],
    "Fatigue / Exhaustion": [
        "fatigue", "exhaustion", "weariness"
    ],
    "Emotional Numbness": [
        "emotional numbness", "detachment", "emotional flatness"
    ],
    "Impatience / Irritability": [
        "impatience", "irritability", "restlessness"
    ],
    "Teasing": [
        "teasing", "playful provocation"
    ],
    "Bittersweetness": [
        "bittersweetness", "mixed joy and sadness"
    ],
    "Sexual Lust": [
        "sexual lust", "sexual desire", "arousal"
    ],
    "Embarrassment": [
        "embarrassment", "awkwardness"
    ],
    "Sourness": [
        "sourness", "negative aftertaste emotion"
    ]
}

# =========================
# Generate combinations
# =========================

output_path = "voice_emotion_grid.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for (
        age,
        softness,
        resonance,
        nasality,
        phonation,
        rate,
        (emotion_category, emotion_words)
    ) in itertools.product(
        ages,
        softness_levels,
        resonance_placements,
        nasality_levels,
        phonation_types,
        speech_rates,
        emotion_taxonomy.items()
    ):
        description = (
            f"Age group: {age}, "
            f"vocal effort: {softness}, "
            f"resonance placement: {resonance}, "
            f"nasality level: {nasality}, "
            f"phonation type: {phonation}, "
            f"speech rate: {rate}, "
            f"emotional expression category: {emotion_category}, "
            f"associated emotion words: {', '.join(emotion_words)}"
        )

        record = {
            "age": age,
            "softness_harshness": softness,
            "resonance": resonance,
            "nasality": nasality,
            "phonation": phonation,
            "speech_rate": rate,
            "emotion_category": emotion_category,
            "emotion_words": emotion_words,
            "description": description
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Generated voice_emotion_grid.jsonl with all combinations.")

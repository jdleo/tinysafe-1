"""Shared utilities for the safety distillation pipeline."""

import json
import hashlib
from pathlib import Path


CATEGORIES = [
    "violence", "hate", "sexual", "self_harm",
    "dangerous_info", "harassment", "illegal_activity",
]

# Mapping from BeaverTails 14 categories → our 7
BEAVERTAILS_CATEGORY_MAP = {
    "animal_abuse": "violence",
    "child_abuse": "violence",
    "controversial_topics,politics": None,  # drop — too noisy
    "discrimination,stereotype,injustice": "hate",
    "drug_abuse,weapons,banned_substance": "dangerous_info",
    "financial_crime,property_crime,theft": "illegal_activity",
    "hate_speech,offensive_language": "hate",
    "misinformation_regarding_ethics,laws_and_safety": "dangerous_info",
    "non_violent_unethical_behavior": None,  # too vague
    "privacy_violation": "harassment",
    "self_harm": "self_harm",
    "sexually_explicit,adult_content": "sexual",
    "terrorism,organized_crime": "violence",
    "violence,aiding_and_abetting,incitement": "violence",
}


def load_config(config_path: str = "configs/config.json") -> dict:
    with open(config_path) as f:
        return json.load(f)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def save_jsonl(data: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_sample(text: str, label: str, categories: dict | None = None, source: str = "") -> dict:
    """Normalize a sample to the unified schema."""
    sample = {
        "text": text.strip(),
        "label": label,  # "safe" or "unsafe"
        "categories": {cat: False for cat in CATEGORIES},
        "source": source,
    }
    if categories:
        for cat, val in categories.items():
            if cat in CATEGORIES:
                sample["categories"][cat] = bool(val)
    return sample

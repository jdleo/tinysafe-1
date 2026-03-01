#!/usr/bin/env python3
"""Usage: uv run infer.py "your text here" """

import sys
from pathlib import Path

import torch
from transformers import DebertaV2Tokenizer

from src.model import SafetyClassifier
from src.utils import CATEGORIES, load_config

config = load_config()
threshold = config.get("inference", {}).get("binary_threshold", 0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])
model = SafetyClassifier(config["base_model"], config["num_categories"])
ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

text = " ".join(sys.argv[1:])
if not text:
    print("Usage: uv run infer.py \"your text here\"")
    sys.exit(1)

inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

result = model.predict(inputs["input_ids"], inputs["attention_mask"])
score = result["unsafe_score"].item()
cat_scores = result["category_scores"].squeeze(0)

label = "UNSAFE" if score > threshold else "SAFE"
print(f"\n  {label}  (score: {score:.3f}, threshold: {threshold})\n")

if score > threshold:
    for i, cat in enumerate(CATEGORIES):
        s = cat_scores[i].item()
        if s > 0.5:
            bar = "█" * int(s * 20)
            print(f"  {cat:<20} {s:.3f}  {bar}")

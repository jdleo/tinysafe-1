#!/usr/bin/env python3
"""Sweep binary threshold on val set to find optimal operating point."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model import SafetyClassifier
from src.dataset import SafetyDataset
from src.utils import load_config

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def main():
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])

    # Load model
    model = SafetyClassifier(config["base_model"], config["num_categories"])
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get val set probabilities
    val_ds = SafetyDataset("data/processed/val.jsonl", tokenizer, config["max_length"])
    use_cuda = device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=128 if use_cuda else 64, shuffle=False,
                            num_workers=4 if use_cuda else 0, pin_memory=use_cuda)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, _ = model(input_ids, attention_mask)
            probs = torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["binary_label"].numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Sweep thresholds
    thresholds = np.arange(0.10, 0.91, 0.05)
    results = []

    print(f"\n{'Thresh':<10} {'F1-M':<10} {'F1-B':<10} {'U-Rec':<10} {'U-Prec':<10} {'FPR':<10}")
    print("-" * 60)

    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = (all_probs > t).astype(int)
        f1_m = f1_score(all_labels, preds, average="macro", zero_division=0)
        f1_b = f1_score(all_labels, preds, average="binary", zero_division=0)
        u_rec = recall_score(all_labels, preds, pos_label=1, zero_division=0)
        u_prec = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        fpr = 1 - recall_score(all_labels, preds, pos_label=0, zero_division=0)

        marker = ""
        if f1_m > best_f1:
            best_f1 = f1_m
            best_threshold = t
            marker = " ★"

        print(f"{t:<10.2f} {f1_m:<10.4f} {f1_b:<10.4f} {u_rec:<10.4f} {u_prec:<10.4f} {fpr:<10.4f}{marker}")
        results.append({
            "threshold": round(float(t), 2),
            "f1_macro": float(f1_m),
            "f1_binary": float(f1_b),
            "unsafe_recall": float(u_rec),
            "unsafe_precision": float(u_prec),
            "fpr": float(fpr),
        })

    print(f"\nBest F1 macro: {best_f1:.4f} at threshold {best_threshold:.2f}")

    # Save results
    output = {
        "best_threshold": round(float(best_threshold), 2),
        "best_f1_macro": float(best_f1),
        "sweep": results,
    }
    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {RESULTS_DIR / 'threshold_sweep.json'}")

    # Also update config with best threshold
    config["inference"] = config.get("inference", {})
    config["inference"]["binary_threshold"] = round(float(best_threshold), 2)
    with open("configs/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated configs/config.json with threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()

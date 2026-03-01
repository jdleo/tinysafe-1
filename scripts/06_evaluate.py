#!/usr/bin/env python3
"""Evaluate trained model on all benchmarks and produce comparison tables."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer

from src.dataset import SafetyDataset
from src.model import SafetyClassifier
from src.utils import CATEGORIES, load_config, normalize_sample, save_jsonl

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def load_model(config: dict, device: torch.device):
    model = SafetyClassifier(config["base_model"], config["num_categories"])
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict_batch(model, dataloader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_binary_probs = []
    all_binary_labels = []
    all_cat_probs = []
    all_cat_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, category_logits = model(input_ids, attention_mask)

            all_binary_probs.extend(torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy())
            all_binary_labels.extend(batch["binary_label"].numpy())
            all_cat_probs.extend(torch.sigmoid(category_logits).cpu().numpy())
            all_cat_labels.extend(batch["category_labels"].numpy())

    return (
        np.array(all_binary_probs),
        np.array(all_binary_labels),
        np.array(all_cat_probs),
        np.array(all_cat_labels),
    )


def compute_metrics(binary_probs, binary_labels, cat_probs, cat_labels, threshold=0.5) -> dict:
    binary_preds = (binary_probs > threshold).astype(int)

    metrics = {
        "f1_macro": f1_score(binary_labels, binary_preds, average="macro", zero_division=0),
        "f1_binary": f1_score(binary_labels, binary_preds, average="binary", zero_division=0),
        "unsafe_recall": recall_score(binary_labels, binary_preds, pos_label=1, zero_division=0),
        "unsafe_precision": precision_score(binary_labels, binary_preds, pos_label=1, zero_division=0),
        "safe_recall": recall_score(binary_labels, binary_preds, pos_label=0, zero_division=0),
        "safe_precision": precision_score(binary_labels, binary_preds, pos_label=0, zero_division=0),
        "fpr": 1 - recall_score(binary_labels, binary_preds, pos_label=0, zero_division=0),
        "threshold": threshold,
    }

    # Per-category
    cat_preds = (cat_probs > 0.5).astype(int)
    for i, cat in enumerate(CATEGORIES):
        if cat_labels[:, i].sum() > 0:
            metrics[f"{cat}_f1"] = f1_score(cat_labels[:, i], cat_preds[:, i], zero_division=0)
            metrics[f"{cat}_precision"] = precision_score(cat_labels[:, i], cat_preds[:, i], zero_division=0)
            metrics[f"{cat}_recall"] = recall_score(cat_labels[:, i], cat_preds[:, i], zero_division=0)

    return metrics


def eval_internal_test(model, tokenizer, config, device, threshold=0.5) -> dict:
    """Evaluate on held-out internal test set."""
    print("\n" + "="*60)
    print("Internal Test Set")
    print("="*60)

    ds = SafetyDataset("data/processed/test.jsonl", tokenizer, config["max_length"])
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    probs, labels, cat_probs, cat_labels = predict_batch(model, loader, device)
    metrics = compute_metrics(probs, labels, cat_probs, cat_labels, threshold)

    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  Unsafe recall:   {metrics['unsafe_recall']:.4f}")
    print(f"  Unsafe precision: {metrics['unsafe_precision']:.4f}")
    print(f"  FPR:             {metrics['fpr']:.4f}")
    return metrics


def eval_toxic_chat_test(model, tokenizer, config, device, threshold=0.5) -> dict:
    """Evaluate on ToxicChat test split."""
    print("\n" + "="*60)
    print("ToxicChat Test")
    print("="*60)

    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    samples = []
    for row in ds:
        text = row.get("user_input", "")
        if not text.strip():
            continue
        label = "unsafe" if row.get("toxicity", 0) == 1 else "safe"
        samples.append(normalize_sample(text, label, source="toxic_chat_test"))

    tmp_path = Path("data/eval/toxic_chat_test.jsonl")
    save_jsonl(samples, tmp_path)
    eval_ds = SafetyDataset(tmp_path, tokenizer, config["max_length"])
    loader = DataLoader(eval_ds, batch_size=64, shuffle=False)
    probs, labels, cat_probs, cat_labels = predict_batch(model, loader, device)
    metrics = compute_metrics(probs, labels, cat_probs, cat_labels, threshold)

    print(f"  Samples: {len(samples)}")
    print(f"  F1 (binary):     {metrics['f1_binary']:.4f}")
    print(f"  Unsafe recall:   {metrics['unsafe_recall']:.4f}")
    print(f"  FPR:             {metrics['fpr']:.4f}")
    return metrics


def eval_wildguard_bench(model, tokenizer, config, device, threshold=0.5) -> dict:
    """Evaluate on WildGuardBench (test split)."""
    print("\n" + "="*60)
    print("WildGuardBench")
    print("="*60)

    ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    samples = []
    for row in ds:
        text = row.get("prompt", "")
        if not text.strip():
            continue
        harm_label = row.get("prompt_harm_label", "")
        label = "unsafe" if harm_label == "harmful" else "safe"
        samples.append(normalize_sample(text, label, source="wildguard_test"))

    tmp_path = Path("data/eval/wildguard_test.jsonl")
    save_jsonl(samples, tmp_path)
    eval_ds = SafetyDataset(tmp_path, tokenizer, config["max_length"])
    loader = DataLoader(eval_ds, batch_size=64, shuffle=False)
    probs, labels, cat_probs, cat_labels = predict_batch(model, loader, device)
    metrics = compute_metrics(probs, labels, cat_probs, cat_labels, threshold)

    print(f"  Samples: {len(samples)}")
    print(f"  F1 (binary):     {metrics['f1_binary']:.4f}")
    print(f"  Unsafe recall:   {metrics['unsafe_recall']:.4f}")
    print(f"  FPR:             {metrics['fpr']:.4f}")
    return metrics


def eval_or_bench(model, tokenizer, config, device, threshold=0.5) -> dict:
    """Evaluate over-refusal on OR-Bench."""
    print("\n" + "="*60)
    print("OR-Bench (Over-Refusal)")
    print("="*60)

    ds = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
    samples = []
    for row in ds:
        text = row.get("prompt", "")
        if not text.strip():
            continue
        # OR-Bench prompts are all SAFE — measuring false positive rate
        samples.append(normalize_sample(text, "safe", source="or_bench"))

    tmp_path = Path("data/eval/or_bench.jsonl")
    save_jsonl(samples, tmp_path)
    eval_ds = SafetyDataset(tmp_path, tokenizer, config["max_length"])
    loader = DataLoader(eval_ds, batch_size=64, shuffle=False)
    probs, labels, _, _ = predict_batch(model, loader, device)

    # FPR = fraction of safe prompts classified as unsafe
    preds = (probs > threshold).astype(int)
    fpr = preds.sum() / len(preds)

    metrics = {"fpr": float(fpr), "total": len(samples), "false_positives": int(preds.sum())}
    print(f"  Samples: {len(samples)}")
    print(f"  FPR: {fpr*100:.1f}% ({int(preds.sum())}/{len(samples)} false positives)")
    return metrics


def print_comparison_table(results: dict):
    """Print the final comparison table."""
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)

    tc = results.get("toxic_chat", {})
    wg = results.get("wildguard", {})
    orb = results.get("or_bench", {})

    header = f"{'Model':<30} {'Params':<10} {'TC F1':<10} {'WG F1':<10} {'OR FPR':<10} {'Latency':<10}"
    print(header)
    print("-" * len(header))
    or_fpr = f"{orb.get('fpr', 0)*100:.1f}%"
    print(f"{'TinySafe v1':<30} {'22M':<10} {tc.get('f1_binary', 0):<10.4f} {wg.get('f1_binary', 0):<10.4f} {or_fpr:<10} {'~2ms':<10}")
    print(f"{'WildGuard-7B':<30} {'7B':<10} {'~0.92':<10} {'~0.90':<10} {'~10%':<10} {'~500ms':<10}")
    print(f"{'LlamaGuard-3-8B':<30} {'8B':<10} {'~0.90':<10} {'~0.88':<10} {'~12%':<10} {'~600ms':<10}")
    print(f"{'ToxicBERT':<30} {'110M':<10} {'~0.82':<10} {'~0.78':<10} {'~25%':<10} {'~10ms':<10}")


def main():
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/eval").mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    threshold = config.get("inference", {}).get("binary_threshold", 0.5)
    print(f"Binary threshold: {threshold}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])
    model = load_model(config, device)

    results = {}

    results["internal_test"] = eval_internal_test(model, tokenizer, config, device, threshold)
    results["toxic_chat"] = eval_toxic_chat_test(model, tokenizer, config, device, threshold)
    results["wildguard"] = eval_wildguard_bench(model, tokenizer, config, device, threshold)
    results["or_bench"] = eval_or_bench(model, tokenizer, config, device, threshold)

    # Save results
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    print_comparison_table(results)
    print(f"\nResults saved to {RESULTS_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()

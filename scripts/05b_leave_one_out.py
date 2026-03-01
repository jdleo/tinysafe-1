#!/usr/bin/env python3
"""Leave-one-dataset-out evaluation for contamination robustness.

For each source dataset, trains on everything EXCEPT that source,
then evaluates on the held-out source. If the model generalizes well
without a source, that's strong evidence the model isn't memorizing
dataset-specific artifacts.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, precision_score

from src.model import SafetyClassifier
from src.dataset import SafetyDataset
from src.losses import DualHeadLoss
from src.utils import CATEGORIES, load_config, load_jsonl, save_jsonl

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
LOO_DIR = Path("data/loo_tmp")


def quick_evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, _ = model(input_ids, attention_mask)
            probs = torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy()
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(batch["binary_label"].numpy().astype(int))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_binary": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "unsafe_recall": recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "unsafe_precision": precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "safe_recall": recall_score(all_labels, all_preds, pos_label=0, zero_division=0),
        "n_samples": len(all_labels),
        "n_unsafe": int(all_labels.sum()),
    }


def train_quick(train_path: Path, config: dict, device: torch.device, tokenizer) -> SafetyClassifier:
    """Abbreviated training run (3 epochs, no early stopping) for LOO."""
    train_config = config["training"]
    model = SafetyClassifier(config["base_model"], config["num_categories"]).to(device)
    criterion = DualHeadLoss(gamma=train_config["focal_loss_gamma"], category_weight=train_config["category_loss_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])

    train_ds = SafetyDataset(train_path, tokenizer, config["max_length"])
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=train_config["batch_size"], shuffle=True,
                              num_workers=4 if use_cuda else 0, pin_memory=use_cuda)

    total_steps = len(train_loader) * 3 // train_config["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * train_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    grad_accum = train_config["gradient_accumulation_steps"]

    for epoch in range(1, 4):  # 3 epochs only
        model.train()
        total_loss = 0
        n = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_labels = batch["binary_label"].to(device)
            category_labels = batch["category_labels"].to(device)

            binary_logits, category_logits = model(input_ids, attention_mask)
            loss_dict = criterion(binary_logits, category_logits, binary_labels, category_labels)
            loss = loss_dict["loss"] / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss_dict["loss"].item()
            n += 1

        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        print(f"    Epoch {epoch}/3 | Loss: {total_loss/n:.4f}")

    return model


def main():
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOO_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])

    # Load full training data (train + val combined for LOO splits)
    train_samples = load_jsonl(PROCESSED_DIR / "train.jsonl")
    val_samples = load_jsonl(PROCESSED_DIR / "val.jsonl")
    all_samples = train_samples + val_samples

    # Get unique sources
    source_counts = Counter(s.get("source", "unknown") for s in all_samples)
    sources = sorted(source_counts.keys())

    print(f"Total samples: {len(all_samples)}")
    print(f"Sources ({len(sources)}):")
    for src, count in source_counts.most_common():
        unsafe = sum(1 for s in all_samples if s.get("source") == src and s["label"] == "unsafe")
        print(f"  {src}: {count} ({unsafe} unsafe)")

    # Leave-one-out for each source
    results = {}
    for held_out_source in sources:
        print(f"\n{'='*60}")
        print(f"LOO: Holding out '{held_out_source}' ({source_counts[held_out_source]} samples)")
        print(f"{'='*60}")

        # Split
        train_data = [s for s in all_samples if s.get("source") != held_out_source]
        eval_data = [s for s in all_samples if s.get("source") == held_out_source]

        if len(eval_data) < 20:
            print(f"  Skipping — only {len(eval_data)} eval samples")
            continue

        unsafe_in_eval = sum(1 for s in eval_data if s["label"] == "unsafe")
        if unsafe_in_eval == 0 or unsafe_in_eval == len(eval_data):
            print(f"  Skipping — eval set is all {'unsafe' if unsafe_in_eval else 'safe'}")
            continue

        # Save temp splits
        train_path = LOO_DIR / f"train_no_{held_out_source}.jsonl"
        eval_path = LOO_DIR / f"eval_{held_out_source}.jsonl"
        save_jsonl(train_data, train_path)
        save_jsonl(eval_data, eval_path)

        print(f"  Train: {len(train_data)} | Eval: {len(eval_data)} ({unsafe_in_eval} unsafe)")

        # Train
        start = time.time()
        model = train_quick(train_path, config, device, tokenizer)
        elapsed = time.time() - start
        print(f"  Training done ({elapsed:.0f}s)")

        # Evaluate
        eval_ds = SafetyDataset(eval_path, tokenizer, config["max_length"])
        eval_loader = DataLoader(eval_ds, batch_size=128 if device.type == "cuda" else 64,
                                  shuffle=False, num_workers=4 if device.type == "cuda" else 0,
                                  pin_memory=device.type == "cuda")
        metrics = quick_evaluate(model, eval_loader, device)

        results[held_out_source] = metrics
        print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
        print(f"  F1 (binary):   {metrics['f1_binary']:.4f}")
        print(f"  Unsafe recall: {metrics['unsafe_recall']:.4f}")
        print(f"  Unsafe prec:   {metrics['unsafe_precision']:.4f}")
        print(f"  Safe recall:   {metrics['safe_recall']:.4f}")

        # Free memory
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("LEAVE-ONE-OUT SUMMARY")
    print(f"{'='*60}")
    header = f"{'Held-Out Source':<25} {'N':<8} {'Unsafe':<8} {'F1-M':<8} {'F1-B':<8} {'U-Rec':<8} {'U-Prec':<8}"
    print(header)
    print("-" * len(header))
    for src, m in sorted(results.items()):
        print(f"{src:<25} {m['n_samples']:<8} {m['n_unsafe']:<8} "
              f"{m['f1_macro']:<8.4f} {m['f1_binary']:<8.4f} "
              f"{m['unsafe_recall']:<8.4f} {m['unsafe_precision']:<8.4f}")

    # Save
    with open(RESULTS_DIR / "leave_one_out_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\nResults saved to {RESULTS_DIR / 'leave_one_out_results.json'}")

    # Interpretation
    f1s = [m["f1_binary"] for m in results.values()]
    if f1s:
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        min_src = min(results, key=lambda s: results[s]["f1_binary"])
        print(f"\nMean F1 across LOO: {mean_f1:.4f} (+/- {std_f1:.4f})")
        print(f"Weakest generalization: '{min_src}' (F1={results[min_src]['f1_binary']:.4f})")
        if std_f1 < 0.05:
            print("Low variance — model generalizes consistently across sources.")
        else:
            print("High variance — model may be over-reliant on specific sources.")


if __name__ == "__main__":
    main()

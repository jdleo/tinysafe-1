#!/usr/bin/env python3
"""Train dual-head DeBERTa safety classifier."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import DebertaV2Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from src.model import SafetyClassifier
from src.dataset import SafetyDataset
from src.losses import DualHeadLoss
from src.utils import CATEGORIES, load_config

# MPS memory config
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

CHECKPOINT_DIR = Path("checkpoints")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def evaluate(model, dataloader, criterion, device) -> dict:
    model.eval()
    total_loss = 0
    all_binary_preds = []
    all_binary_labels = []
    all_cat_preds = []
    all_cat_labels = []
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_labels = batch["binary_label"].to(device)
            category_labels = batch["category_labels"].to(device)

            binary_logits, category_logits = model(input_ids, attention_mask)
            loss_dict = criterion(binary_logits, category_logits, binary_labels, category_labels)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

            binary_probs = torch.sigmoid(binary_logits.squeeze(-1))
            binary_preds = (binary_probs > 0.5).cpu().numpy()
            all_binary_preds.extend(binary_preds)
            all_binary_labels.extend(binary_labels.cpu().numpy())

            cat_probs = torch.sigmoid(category_logits)
            cat_preds = (cat_probs > 0.5).cpu().numpy()
            all_cat_preds.extend(cat_preds)
            all_cat_labels.extend(category_labels.cpu().numpy())

    all_binary_preds = np.array(all_binary_preds)
    all_binary_labels = np.array(all_binary_labels)
    all_cat_preds = np.array(all_cat_preds)
    all_cat_labels = np.array(all_cat_labels)

    # Binary metrics
    unsafe_mask = all_binary_labels == 1
    safe_mask = all_binary_labels == 0

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "f1_macro": f1_score(all_binary_labels, all_binary_preds, average="macro", zero_division=0),
        "f1_binary": f1_score(all_binary_labels, all_binary_preds, average="binary", zero_division=0),
        "unsafe_recall": recall_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0),
        "unsafe_precision": precision_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0),
        "safe_recall": recall_score(all_binary_labels, all_binary_preds, pos_label=0, zero_division=0),
        "safe_precision": precision_score(all_binary_labels, all_binary_preds, pos_label=0, zero_division=0),
    }

    # Per-category metrics
    for i, cat in enumerate(CATEGORIES):
        cat_labels_i = all_cat_labels[:, i]
        cat_preds_i = all_cat_preds[:, i]
        if cat_labels_i.sum() > 0:
            metrics[f"{cat}_recall"] = recall_score(cat_labels_i, cat_preds_i, zero_division=0)
            metrics[f"{cat}_precision"] = precision_score(cat_labels_i, cat_preds_i, zero_division=0)
            metrics[f"{cat}_f1"] = f1_score(cat_labels_i, cat_preds_i, zero_division=0)

    return metrics


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, grad_accum_steps) -> float:
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        binary_labels = batch["binary_label"].to(device)
        category_labels = batch["category_labels"].to(device)

        binary_logits, category_logits = model(input_ids, attention_mask)
        loss_dict = criterion(binary_logits, category_logits, binary_labels, category_labels)
        loss = loss_dict["loss"] / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss_dict["loss"].item()
        num_batches += 1

        if (step + 1) % 100 == 0:
            avg = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            print(f"    Step {step+1}/{len(dataloader)} | Loss: {avg:.4f} | LR: {lr:.2e}")

    # Handle remaining gradients
    if len(dataloader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(num_batches, 1)


def main():
    config = load_config()
    train_config = config["training"]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Tokenizer
    print(f"Loading tokenizer: {config['base_model']}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])

    # Datasets
    print("Loading datasets...")
    train_ds = SafetyDataset("data/processed/train.jsonl", tokenizer, config["max_length"])
    val_ds = SafetyDataset("data/processed/val.jsonl", tokenizer, config["max_length"])

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    print(f"Loading model: {config['base_model']}")
    model = SafetyClassifier(config["base_model"], config["num_categories"])
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} total, {trainable:,} trainable")

    # Loss, optimizer, scheduler
    criterion = DualHeadLoss(
        gamma=train_config["focal_loss_gamma"],
        category_weight=train_config["category_loss_weight"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    total_steps = len(train_loader) * train_config["num_epochs"] // train_config["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * train_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_metric = 0.0
    patience_counter = 0
    best_model_metric = train_config["best_model_metric"]
    patience = train_config["early_stopping_patience"]

    print(f"\nTraining for {train_config['num_epochs']} epochs")
    print(f"Best model metric: {best_model_metric}")
    print(f"Early stopping patience: {patience}")
    print(f"Effective batch size: {train_config['batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"Total steps: {total_steps} (warmup: {warmup_steps})")
    print()

    for epoch in range(1, train_config["num_epochs"] + 1):
        start = time.time()
        print(f"Epoch {epoch}/{train_config['num_epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, train_config["gradient_accumulation_steps"],
        )
        elapsed = time.time() - start
        print(f"  Train loss: {train_loss:.4f} ({elapsed:.1f}s)")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"  Val loss:        {val_metrics['loss']:.4f}")
        print(f"  Val F1 (macro):  {val_metrics['f1_macro']:.4f}")
        print(f"  Val F1 (binary): {val_metrics['f1_binary']:.4f}")
        print(f"  Unsafe recall:   {val_metrics['unsafe_recall']:.4f}")
        print(f"  Unsafe prec:     {val_metrics['unsafe_precision']:.4f}")
        print(f"  Safe recall:     {val_metrics['safe_recall']:.4f}")

        # Per-category
        print("  Categories:")
        for cat in CATEGORIES:
            f1_key = f"{cat}_f1"
            if f1_key in val_metrics:
                print(f"    {cat}: F1={val_metrics[f1_key]:.3f} "
                      f"P={val_metrics[f'{cat}_precision']:.3f} "
                      f"R={val_metrics[f'{cat}_recall']:.3f}")

        # Early stopping
        current_metric = val_metrics[best_model_metric]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  ★ New best {best_model_metric}: {best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        }, CHECKPOINT_DIR / "latest_model.pt")

    # Save final metrics
    with open(CHECKPOINT_DIR / "training_metrics.json", "w") as f:
        json.dump({"best_metric": best_metric, "best_metric_name": best_model_metric}, f, indent=2)

    print(f"\nTraining complete. Best {best_model_metric}: {best_metric:.4f}")
    print(f"Best model saved to {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()

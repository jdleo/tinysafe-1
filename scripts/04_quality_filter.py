#!/usr/bin/env python3
"""Filter, deduplicate, balance, and split the labeled dataset."""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.utils import CATEGORIES, load_config, load_jsonl, save_jsonl

DATA_DIR = Path("data")
LABELED_DIR = DATA_DIR / "labeled"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"


def _label_counts(samples: list[dict]) -> str:
    safe = sum(1 for s in samples if s.get("label") == "safe")
    unsafe = len(samples) - safe
    return f"[safe={safe}, unsafe={unsafe}]"


def build_eval_hashes() -> set[str]:
    """Load all eval benchmark texts and build a hash set for contamination checking."""
    from datasets import load_dataset

    eval_hashes = set()
    eval_texts = []

    # ToxicChat test
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        for row in ds:
            text = row.get("user_input", "")
            if text.strip():
                eval_texts.append(text)
    except Exception as e:
        print(f"  Warning: could not load ToxicChat test: {e}")

    # WildGuardBench test
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        for row in ds:
            text = row.get("prompt", "") or row.get("instruction", "") or ""
            if text.strip():
                eval_texts.append(text)
    except Exception as e:
        print(f"  Warning: could not load WildGuardBench test: {e}")

    # OR-Bench
    try:
        ds = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
        for row in ds:
            text = row.get("prompt", "")
            if text.strip():
                eval_texts.append(text)
    except Exception as e:
        print(f"  Warning: could not load OR-Bench: {e}")

    for text in eval_texts:
        eval_hashes.add(hashlib.sha256(text.strip().lower().encode()).hexdigest())

    print(f"  Built eval hash set: {len(eval_hashes)} unique eval texts")
    return eval_hashes


def contamination_filter(samples: list[dict], eval_hashes: set[str]) -> list[dict]:
    """Remove any training samples that appear in eval benchmarks (exact match)."""
    before = len(samples)
    filtered = []
    contaminated = 0
    for s in samples:
        h = hashlib.sha256(s["text"].strip().lower().encode()).hexdigest()
        if h in eval_hashes:
            contaminated += 1
        else:
            filtered.append(s)
    print(f"  Contamination filter: {before} → {len(filtered)} (removed {contaminated} eval overlaps) {_label_counts(filtered)}")
    return filtered


def confidence_filter(samples: list[dict], min_confidence: float) -> list[dict]:
    """Drop samples below confidence threshold."""
    before = len(samples)
    filtered = [s for s in samples if s.get("claude_confidence", 1.0) >= min_confidence]
    print(f"  Confidence filter (>={min_confidence}): {before} → {len(filtered)} (dropped {before - len(filtered)}) {_label_counts(filtered)}")
    return filtered


def exact_dedup(samples: list[dict]) -> list[dict]:
    """Hash-based exact deduplication on normalized text."""
    before = len(samples)
    seen = set()
    deduped = []
    for s in samples:
        h = hashlib.sha256(s["text"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    print(f"  Exact dedup: {before} → {len(deduped)} (dropped {before - len(deduped)}) {_label_counts(deduped)}")
    return deduped


def near_dedup(samples: list[dict], threshold: float = 0.95) -> list[dict]:
    """MinHash-based near-deduplication for fuzzy matches."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  Near dedup: skipped (datasketch not installed)")
        return samples

    before = len(samples)
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = []

    for i, s in enumerate(tqdm(samples, desc="  Computing MinHashes")):
        m = MinHash(num_perm=128)
        words = s["text"].strip().lower().split()
        for w in words:
            m.update(w.encode("utf-8"))
        minhashes.append(m)
        try:
            lsh.insert(f"s-{i}", m)
        except ValueError:
            pass  # duplicate detected by LSH

    # Find duplicates
    to_remove = set()
    for i, m in enumerate(minhashes):
        if i in to_remove:
            continue
        neighbors = lsh.query(m)
        for n in neighbors:
            idx = int(n.split("-")[1])
            if idx != i and idx not in to_remove:
                to_remove.add(idx)

    deduped = [s for i, s in enumerate(samples) if i not in to_remove]
    print(f"  Near dedup (>{threshold}): {before} → {len(deduped)} (dropped {before - len(deduped)}) {_label_counts(deduped)}")
    return deduped


def length_filter(samples: list[dict], min_tokens: int, max_tokens: int) -> list[dict]:
    """Drop samples outside token length bounds (approximate by whitespace split).

    Uses min_tokens=3 as a floor regardless of config, since many safety prompts
    are short (e.g., 'how to kill someone') and whitespace-split undercounts vs
    actual BPE tokens.
    """
    effective_min = min(min_tokens, 3)  # short prompts are valid safety data
    before = len(samples)
    filtered = [s for s in samples if effective_min <= len(s["text"].split()) <= max_tokens]
    print(f"  Length filter ({effective_min}-{max_tokens} words): {before} → {len(filtered)} (dropped {before - len(filtered)}) {_label_counts(filtered)}")
    return filtered


def class_balance(samples: list[dict], safe_ratio: float, unsafe_ratio: float) -> list[dict]:
    """Downsample majority class to approximate target ratio.

    Keeps ALL minority-class samples and downsamples majority class to match
    the target ratio. This prevents throwing away precious unsafe examples.
    """
    random.seed(42)
    safe = [s for s in samples if s["label"] == "safe"]
    unsafe = [s for s in samples if s["label"] == "unsafe"]

    # Keep all of the minority class, downsample majority to match ratio
    if len(unsafe) == 0 or len(safe) == 0:
        print(f"  Class balance: skipped (safe={len(safe)}, unsafe={len(unsafe)})")
        return samples

    # Desired: safe/unsafe = safe_ratio/unsafe_ratio
    # Keep all unsafe, compute how many safe we need
    target_safe = int(len(unsafe) * safe_ratio / unsafe_ratio)

    if len(safe) > target_safe:
        safe = random.sample(safe, target_safe)
        print(f"  Class balance: downsampled safe {len(samples) - len(unsafe)} → {len(safe)}")
    else:
        print(f"  Class balance: not enough safe samples to downsample (have {len(safe)}, target {target_safe})")

    balanced = safe + unsafe
    random.shuffle(balanced)
    print(f"  Class balance: {len(safe)} safe ({len(safe)/len(balanced)*100:.1f}%), "
          f"{len(unsafe)} unsafe ({len(unsafe)/len(balanced)*100:.1f}%)")
    return balanced


def stratified_split(samples: list[dict], train_ratio: float, val_ratio: float, test_ratio: float):
    """Split with stratification by source and label."""
    random.seed(42)

    # Group by (source, label)
    groups = {}
    for s in samples:
        key = (s.get("source", "unknown"), s["label"])
        groups.setdefault(key, []).append(s)

    train, val, test = [], [], []
    for key, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main():
    config = load_config()
    filter_config = config["filtering"]
    split_config = config["splits"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load labeled data
    labeled_file = LABELED_DIR / "final_labeled.jsonl"
    if not labeled_file.exists():
        print(f"ERROR: {labeled_file} not found. Run 03_label_with_claude.py first.")
        sys.exit(1)

    samples = load_jsonl(labeled_file)
    print(f"Loaded {len(samples)} labeled samples\n")

    # Build eval hash set for contamination check
    print("Building eval benchmark hash set...")
    eval_hashes = build_eval_hashes()

    # Sequential filters
    print("\nApplying filters:")
    samples = contamination_filter(samples, eval_hashes)
    samples = confidence_filter(samples, filter_config["min_confidence"])
    samples = exact_dedup(samples)
    samples = near_dedup(samples, filter_config["dedup_similarity_threshold"])
    samples = length_filter(samples, filter_config["min_tokens"], filter_config["max_tokens"])
    samples = class_balance(samples, filter_config["target_safe_ratio"], filter_config["target_unsafe_ratio"])

    print(f"\nAfter filtering: {len(samples)} samples")

    # Split
    print("\nSplitting...")
    train, val, test = stratified_split(
        samples,
        split_config["train"],
        split_config["val"],
        split_config["test"],
    )

    save_jsonl(train, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val, PROCESSED_DIR / "val.jsonl")
    save_jsonl(test, PROCESSED_DIR / "test.jsonl")

    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    # Category distribution
    print("\nCategory distribution (train):")
    for cat in CATEGORIES:
        count = sum(1 for s in train if s.get("categories", {}).get(cat, False))
        print(f"  {cat}: {count} ({count/len(train)*100:.1f}%)")

    # Source distribution
    print("\nSource distribution (train):")
    source_counts = Counter(s.get("source", "unknown") for s in train)
    for src, count in source_counts.most_common():
        print(f"  {src}: {count} ({count/len(train)*100:.1f}%)")


if __name__ == "__main__":
    main()

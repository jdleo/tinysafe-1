#!/usr/bin/env python3
"""Download and normalize all public datasets into unified JSONL format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from src.utils import (
    BEAVERTAILS_CATEGORY_MAP,
    CATEGORIES,
    load_config,
    normalize_sample,
    save_jsonl,
)

DATA_DIR = Path("data/raw")


def download_wildguard(max_samples: int = 30_000) -> list[dict]:
    print("Downloading WildGuardMix (train)...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    samples = []
    for row in tqdm(ds, desc="WildGuard"):
        if len(samples) >= max_samples:
            break
        text = row.get("prompt", "") or row.get("instruction", "") or ""
        if not text.strip():
            continue
        is_harmful = row.get("is_harmful", row.get("harm_label", ""))
        if isinstance(is_harmful, str):
            label = "unsafe" if is_harmful.lower() in ("yes", "harmful", "true", "1", "unsafe") else "safe"
        else:
            label = "unsafe" if is_harmful else "safe"
        # WildGuard has harm_category sometimes
        cats = {cat: False for cat in CATEGORIES}
        harm_cat = row.get("harm_category", None)
        if harm_cat and label == "unsafe":
            harm_cat_lower = str(harm_cat).lower()
            if "violen" in harm_cat_lower:
                cats["violence"] = True
            if "hate" in harm_cat_lower or "discriminat" in harm_cat_lower:
                cats["hate"] = True
            if "sexual" in harm_cat_lower:
                cats["sexual"] = True
            if "self" in harm_cat_lower and "harm" in harm_cat_lower:
                cats["self_harm"] = True
            if "danger" in harm_cat_lower or "weapon" in harm_cat_lower or "drug" in harm_cat_lower:
                cats["dangerous_info"] = True
            if "harass" in harm_cat_lower:
                cats["harassment"] = True
            if "illegal" in harm_cat_lower or "crime" in harm_cat_lower:
                cats["illegal_activity"] = True
        samples.append(normalize_sample(text, label, cats, source="wildguard"))
    print(f"  → {len(samples)} samples ({sum(1 for s in samples if s['label']=='unsafe')} unsafe)")
    return samples


def download_beavertails(max_samples: int = 30_000) -> list[dict]:
    print("Downloading BeaverTails...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    samples = []
    for row in tqdm(ds, desc="BeaverTails"):
        if len(samples) >= max_samples:
            break
        text = row.get("prompt", "")
        if not text.strip():
            continue
        is_safe = row.get("is_safe", True)
        label = "safe" if is_safe else "unsafe"
        cats = {cat: False for cat in CATEGORIES}
        # Map BeaverTails categories
        bt_cats = row.get("category", {})
        if isinstance(bt_cats, dict):
            for bt_cat, is_present in bt_cats.items():
                if is_present and bt_cat in BEAVERTAILS_CATEGORY_MAP:
                    mapped = BEAVERTAILS_CATEGORY_MAP[bt_cat]
                    if mapped:
                        cats[mapped] = True
        samples.append(normalize_sample(text, label, cats, source="beavertails"))
    print(f"  → {len(samples)} samples ({sum(1 for s in samples if s['label']=='unsafe')} unsafe)")
    return samples


def download_toxigen(max_samples: int = 15_000) -> list[dict]:
    print("Downloading ToxiGen (annotated)...")
    ds = load_dataset("skg/toxigen-data", "annotated", split="train")
    samples = []
    for row in tqdm(ds, desc="ToxiGen"):
        if len(samples) >= max_samples:
            break
        text = row.get("text", "")
        if not text.strip():
            continue
        # ToxiGen uses toxicity_ai and toxicity_human scores
        tox_score = row.get("toxicity_ai", row.get("toxicity_human", 0))
        if isinstance(tox_score, (int, float)):
            label = "unsafe" if tox_score >= 0.5 else "safe"
        else:
            label = "unsafe" if str(tox_score).lower() in ("toxic", "1", "true") else "safe"
        cats = {cat: False for cat in CATEGORIES}
        if label == "unsafe":
            cats["hate"] = True  # ToxiGen is primarily implicit hate speech
        samples.append(normalize_sample(text, label, cats, source="toxigen"))
    print(f"  → {len(samples)} samples ({sum(1 for s in samples if s['label']=='unsafe')} unsafe)")
    return samples


def download_toxic_chat(max_samples: int = 7_000) -> list[dict]:
    print("Downloading Toxic-Chat (train)...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    samples = []
    for row in tqdm(ds, desc="ToxicChat"):
        if len(samples) >= max_samples:
            break
        text = row.get("user_input", "")
        if not text.strip():
            continue
        toxicity = row.get("toxicity", 0)
        label = "unsafe" if toxicity == 1 else "safe"
        cats = {cat: False for cat in CATEGORIES}
        # Toxic-Chat doesn't have fine-grained categories — leave for Claude labeling
        samples.append(normalize_sample(text, label, cats, source="toxic_chat"))
    print(f"  → {len(samples)} samples ({sum(1 for s in samples if s['label']=='unsafe')} unsafe)")
    return samples


def download_xstest() -> list[dict]:
    print("Downloading XSTest...")
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    samples = []
    for row in tqdm(ds, desc="XSTest"):
        text = row.get("prompt", "")
        if not text.strip():
            continue
        # Types starting with "contrast_" are unsafe counterparts; the rest are safe-but-tricky
        prompt_type = row.get("type", "")
        label = "unsafe" if prompt_type.startswith("contrast_") else "safe"
        samples.append(normalize_sample(text, label, source="xstest"))
    print(f"  → {len(samples)} samples ({sum(1 for s in samples if s['label']=='unsafe')} unsafe)")
    return samples


def download_harmbench() -> list[dict]:
    print("Downloading HarmBench...")
    samples = []
    for config in ["DirectRequest", "HumanJailbreaks"]:
        for split in ["val", "test"]:
            ds = load_dataset("mariagrandury/harmbench", config, split=split)
            for row in ds:
                text = row.get("Behavior", "")
                if not text.strip():
                    continue
                cats = {cat: False for cat in CATEGORIES}
                semantic_cat = str(row.get("SemanticCategory", "")).lower()
                if "violen" in semantic_cat:
                    cats["violence"] = True
                if "hate" in semantic_cat or "discriminat" in semantic_cat:
                    cats["hate"] = True
                if "sex" in semantic_cat:
                    cats["sexual"] = True
                if "self" in semantic_cat:
                    cats["self_harm"] = True
                if "cyber" in semantic_cat or "weapon" in semantic_cat or "chem" in semantic_cat:
                    cats["dangerous_info"] = True
                if "harass" in semantic_cat:
                    cats["harassment"] = True
                if "illegal" in semantic_cat or "fraud" in semantic_cat:
                    cats["illegal_activity"] = True
                samples.append(normalize_sample(text, "unsafe", cats, source="harmbench"))
    print(f"  → {len(samples)} samples (all unsafe)")
    return samples


def download_sorry_bench() -> list[dict]:
    print("Downloading SORRY-Bench...")
    ds = load_dataset("sorry-bench/sorry-bench-202406", split="train")
    samples = []
    for row in tqdm(ds, desc="SORRY-Bench"):
        text = row.get("turns", [""])[0] if isinstance(row.get("turns"), list) else row.get("prompt", "")
        if not text or not text.strip():
            continue
        cats = {cat: False for cat in CATEGORIES}
        category = str(row.get("category", "")).lower()
        if "violen" in category:
            cats["violence"] = True
        if "hate" in category or "discriminat" in category:
            cats["hate"] = True
        if "sex" in category:
            cats["sexual"] = True
        if "self" in category and "harm" in category:
            cats["self_harm"] = True
        if "danger" in category or "weapon" in category:
            cats["dangerous_info"] = True
        if "harass" in category:
            cats["harassment"] = True
        if "illegal" in category:
            cats["illegal_activity"] = True
        samples.append(normalize_sample(text, "unsafe", cats, source="sorry_bench"))
    print(f"  → {len(samples)} samples (all unsafe)")
    return samples


def main():
    config = load_config()

    all_samples = []

    # Core training data
    all_samples.extend(download_wildguard(30_000))
    all_samples.extend(download_beavertails(30_000))
    all_samples.extend(download_toxigen(15_000))
    all_samples.extend(download_toxic_chat(7_000))

    # Critical supplements
    all_samples.extend(download_xstest())
    all_samples.extend(download_harmbench())
    all_samples.extend(download_sorry_bench())

    # Save combined raw dataset
    save_jsonl(all_samples, DATA_DIR / "all_public.jsonl")

    # Summary
    total = len(all_samples)
    unsafe = sum(1 for s in all_samples if s["label"] == "unsafe")
    print(f"\nTotal: {total} samples")
    print(f"  Safe:   {total - unsafe} ({(total-unsafe)/total*100:.1f}%)")
    print(f"  Unsafe: {unsafe} ({unsafe/total*100:.1f}%)")

    by_source = {}
    for s in all_samples:
        src = s["source"]
        by_source.setdefault(src, {"safe": 0, "unsafe": 0})
        by_source[src][s["label"]] += 1
    print("\nBy source:")
    for src, counts in sorted(by_source.items()):
        print(f"  {src}: {counts['safe']} safe, {counts['unsafe']} unsafe")


if __name__ == "__main__":
    main()

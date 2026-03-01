#!/usr/bin/env python3
"""Generate synthetic training data via Claude Haiku (Anthropic API)."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
from tqdm import tqdm

from src.utils import CATEGORIES, load_config, normalize_sample, save_jsonl

DATA_DIR = Path("data/raw")

# Haiku pricing (per token)
INPUT_COST_PER_TOKEN = 0.80 / 1_000_000   # $0.80/MTok
OUTPUT_COST_PER_TOKEN = 4.00 / 1_000_000   # $4.00/MTok

UNSAFE_PROMPT = """Generate 10 diverse examples of text containing {category} content that a safety classifier should flag. Vary the subtlety level.

Return ONLY a JSON array:
[{{"text": "...", "subtlety": "obvious|subtle|coded|edge_case"}}]

Requirements:
- "obvious": Clearly and explicitly unsafe
- "subtle": Unsafe intent but no explicit language
- "coded": Uses slang, dogwhistles, or euphemisms
- "edge_case": Borderline — reasonable people might disagree

Make examples realistic — things real users might actually type into a chatbot. Vary length from one sentence to a short paragraph."""

SAFE_TRICKY_PROMPT = """Generate 10 examples of SAFE text that a naive safety classifier might incorrectly flag as {category}. These are legitimate, harmless uses of sensitive topics.

Return ONLY a JSON array:
[{{"text": "...", "context": "educational|journalistic|academic|fiction|medical|legal"}}]

Topics to draw from: war history, crime reporting, medical drug information, self-defense discussion, sexual health education, legal analysis of criminal cases, fiction with conflict, chemistry/biology education, cybersecurity education, martial arts, hunting/outdoor sports.

Make examples realistic and varied in length."""


class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()

    def add(self, usage):
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_requests += 1

    def add_failure(self):
        self.failed_requests += 1

    @property
    def cost(self) -> float:
        return (self.total_input_tokens * INPUT_COST_PER_TOKEN +
                self.total_output_tokens * OUTPUT_COST_PER_TOKEN)

    def report(self, prefix: str = ""):
        elapsed = time.time() - self.start_time
        mins = elapsed / 60
        print(f"{prefix}  Cost: ${self.cost:.4f} | "
              f"Requests: {self.total_requests} ({self.failed_requests} failed) | "
              f"Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out | "
              f"Time: {mins:.1f}m")


def generate_batch(client: anthropic.Anthropic, prompt: str, model: str, tracker: CostTracker) -> list[dict]:
    """Call Anthropic API and parse the JSON array response."""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            tracker.add(response.usage)
            content = response.content[0].text.strip()
            # Handle markdown-wrapped JSON
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            tracker.add_failure()
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"  Failed after 3 attempts: {e}")
            return []


def generate_unsafe(client: anthropic.Anthropic, model: str, num_per_category: int, tracker: CostTracker) -> list[dict]:
    """Generate unsafe examples across all categories."""
    samples = []
    batches_per_cat = num_per_category // 10

    for cat_idx, category in enumerate(CATEGORIES):
        cat_display = category.replace("_", " ")
        print(f"  [{cat_idx+1}/{len(CATEGORIES)}] Generating unsafe/{cat_display}...")
        for _ in tqdm(range(batches_per_cat), desc=f"  {cat_display}"):
            prompt = UNSAFE_PROMPT.format(category=cat_display)
            results = generate_batch(client, prompt, model, tracker)
            for item in results:
                if "text" in item:
                    cats = {c: (c == category) for c in CATEGORIES}
                    samples.append(normalize_sample(
                        item["text"], "unsafe", cats, source="synthetic_unsafe",
                    ))
            time.sleep(0.5)  # rate limit courtesy
        tracker.report(f"  [{cat_idx+1}/{len(CATEGORIES)}]")

    print(f"  → {len(samples)} unsafe synthetic samples")
    return samples


def generate_safe_tricky(client: anthropic.Anthropic, model: str, num_per_category: int, tracker: CostTracker) -> list[dict]:
    """Generate safe-but-tricky examples that could trigger false positives."""
    samples = []
    batches_per_cat = num_per_category // 10

    for cat_idx, category in enumerate(CATEGORIES):
        cat_display = category.replace("_", " ")
        print(f"  [{cat_idx+1}/{len(CATEGORIES)}] Generating safe-tricky/{cat_display}...")
        for _ in tqdm(range(batches_per_cat), desc=f"  {cat_display}"):
            prompt = SAFE_TRICKY_PROMPT.format(category=cat_display)
            results = generate_batch(client, prompt, model, tracker)
            for item in results:
                if "text" in item:
                    samples.append(normalize_sample(
                        item["text"], "safe", source="synthetic_safe_tricky",
                    ))
            time.sleep(0.5)
        tracker.report(f"  [{cat_idx+1}/{len(CATEGORIES)}]")

    print(f"  → {len(samples)} safe-tricky synthetic samples")
    return samples


def main():
    config = load_config()
    syn_config = config["synthetic"]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    model = syn_config["model"]
    tracker = CostTracker()

    print(f"Using model: {model}")
    print(f"Target: {syn_config['num_unsafe']} unsafe + {syn_config['num_safe_tricky']} safe-tricky")
    print(f"Pricing: ${INPUT_COST_PER_TOKEN*1e6:.2f}/MTok in, ${OUTPUT_COST_PER_TOKEN*1e6:.2f}/MTok out\n")

    # Per-category counts
    unsafe_per_cat = syn_config["num_unsafe"] // len(CATEGORIES)
    safe_per_cat = syn_config["num_safe_tricky"] // len(CATEGORIES)

    print("=== Phase 1: Unsafe examples ===")
    unsafe_samples = generate_unsafe(client, model, unsafe_per_cat, tracker)
    print()
    tracker.report("After unsafe gen:")
    print()

    print("=== Phase 2: Safe-tricky examples ===")
    safe_samples = generate_safe_tricky(client, model, safe_per_cat, tracker)

    all_synthetic = unsafe_samples + safe_samples
    save_jsonl(all_synthetic, DATA_DIR / "synthetic.jsonl")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Total synthetic: {len(all_synthetic)}")
    print(f"  Unsafe: {len(unsafe_samples)}")
    print(f"  Safe:   {len(safe_samples)}")
    tracker.report("Final:")
    print(f"Saved to {DATA_DIR / 'synthetic.jsonl'}")


if __name__ == "__main__":
    main()

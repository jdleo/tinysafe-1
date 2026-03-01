#!/usr/bin/env python3
"""Label all data through Claude Batch API for consistent safety labels."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
from tqdm import tqdm

from src.utils import CATEGORIES, load_config, load_jsonl, save_jsonl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"

# Batch API pricing (50% discount on standard)
HAIKU_INPUT_PER_TOKEN = 0.40 / 1_000_000    # $0.40/MTok (batch)
HAIKU_OUTPUT_PER_TOKEN = 2.00 / 1_000_000   # $2.00/MTok (batch)
SONNET_INPUT_PER_TOKEN = 1.50 / 1_000_000   # $1.50/MTok (batch)
SONNET_OUTPUT_PER_TOKEN = 7.50 / 1_000_000  # $7.50/MTok (batch)

SYSTEM_PROMPT = """You are a content safety classifier. Your job is to determine if text is safe or unsafe.

UNSAFE means: the text could cause real-world harm, promotes or instructs violence, contains hate speech targeting protected groups, includes sexual exploitation content, encourages self-harm, provides dangerous instructions (weapons, drugs, hacking), constitutes harassment, or facilitates illegal activity.

SAFE means: everything else. This explicitly includes:
- Academic/educational discussion of unsafe topics
- Journalism and news reporting
- Fiction and creative writing with dark themes
- Medical/legal/scientific information
- Edgy humor that doesn't cross into harm
- Historical discussion of atrocities
- Self-defense information

When in doubt, lean SAFE. Over-refusal is a failure mode."""

USER_PROMPT_TEMPLATE = """Classify this text. Respond with ONLY valid JSON.

{{
  "label": "safe" or "unsafe",
  "categories": {{
    "violence": true/false,
    "hate": true/false,
    "sexual": true/false,
    "self_harm": true/false,
    "dangerous_info": true/false,
    "harassment": true/false,
    "illegal_activity": true/false
  }},
  "confidence": 0.0-1.0
}}

Text: {text}"""


class BatchCostTracker:
    def __init__(self):
        self.batches = []  # list of {model, input_tokens, output_tokens, cost, samples}
        self.start_time = time.time()

    def add_batch(self, model: str, results: list, num_samples: int):
        input_tokens = 0
        output_tokens = 0
        for r in results:
            if r.result.type == "succeeded":
                usage = r.result.message.usage
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens

        is_sonnet = "sonnet" in model.lower()
        input_rate = SONNET_INPUT_PER_TOKEN if is_sonnet else HAIKU_INPUT_PER_TOKEN
        output_rate = SONNET_OUTPUT_PER_TOKEN if is_sonnet else HAIKU_OUTPUT_PER_TOKEN
        cost = input_tokens * input_rate + output_tokens * output_rate

        self.batches.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "samples": num_samples,
        })

        return cost

    @property
    def total_cost(self) -> float:
        return sum(b["cost"] for b in self.batches)

    @property
    def total_input_tokens(self) -> int:
        return sum(b["input_tokens"] for b in self.batches)

    @property
    def total_output_tokens(self) -> int:
        return sum(b["output_tokens"] for b in self.batches)

    def report(self, prefix: str = ""):
        elapsed = time.time() - self.start_time
        mins = elapsed / 60
        print(f"{prefix}  Total cost: ${self.total_cost:.4f} | "
              f"Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out | "
              f"Batches: {len(self.batches)} | Time: {mins:.1f}m")
        for i, b in enumerate(self.batches):
            print(f"    Batch {i+1}: {b['model']} | {b['samples']} samples | "
                  f"${b['cost']:.4f} | {b['input_tokens']:,} in / {b['output_tokens']:,} out")


def create_batch_requests(samples: list[dict], model: str) -> list[dict]:
    """Create Batch API request objects."""
    requests = []
    for i, sample in enumerate(samples):
        text = sample["text"][:2000]  # truncate very long texts for labeling
        requests.append({
            "custom_id": f"sample-{i}",
            "params": {
                "model": model,
                "max_tokens": 256,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
                ],
            },
        })
    return requests


def submit_batch(client: anthropic.Anthropic, requests: list[dict]) -> str:
    """Submit a batch and return the batch ID."""
    batch_file = LABELED_DIR / "batch_requests.jsonl"
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.processing_status}")
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str, model: str, num_samples: int, poll_interval: int = 60) -> list[dict]:
    """Poll batch status until complete, then retrieve results."""
    print(f"Waiting for batch {batch_id}...")
    is_sonnet = "sonnet" in model.lower()
    input_rate = SONNET_INPUT_PER_TOKEN if is_sonnet else HAIKU_INPUT_PER_TOKEN
    output_rate = SONNET_OUTPUT_PER_TOKEN if is_sonnet else HAIKU_OUTPUT_PER_TOKEN
    # Estimate cost based on expected ~320 input + ~40 output tokens per sample
    est_cost = num_samples * (320 * input_rate + 40 * output_rate)
    print(f"Estimated cost for this batch: ~${est_cost:.2f}")

    poll_start = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        elapsed = (time.time() - poll_start) / 60
        pct = counts.succeeded / total * 100 if total > 0 else 0
        print(f"  [{elapsed:5.1f}m] {batch.processing_status} | "
              f"Done: {counts.succeeded}/{total} ({pct:.0f}%) | "
              f"Errors: {counts.errored} | "
              f"Est. cost: ~${est_cost:.2f}")

        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)

    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(result)
    return results


def parse_batch_results(results: list, samples: list[dict]) -> list[dict]:
    """Parse batch results and merge labels into samples."""
    labeled = []
    parse_errors = 0

    result_map = {}
    for r in results:
        result_map[r.custom_id] = r

    for i, sample in enumerate(samples):
        custom_id = f"sample-{i}"
        r = result_map.get(custom_id)

        if not r or r.result.type != "succeeded":
            parse_errors += 1
            continue

        try:
            content = r.result.message.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)

            sample["label"] = parsed["label"]
            sample["categories"] = {
                cat: parsed.get("categories", {}).get(cat, False)
                for cat in CATEGORIES
            }
            sample["claude_confidence"] = parsed.get("confidence", 0.0)
            labeled.append(sample)
        except (json.JSONDecodeError, KeyError, IndexError):
            parse_errors += 1

    print(f"Parsed {len(labeled)}/{len(samples)} ({parse_errors} errors)")
    return labeled


def run_sonnet_qa(client: anthropic.Anthropic, samples: list[dict], model: str, sample_rate: float, tracker: BatchCostTracker) -> list[dict]:
    """Run Sonnet QA on a subset and filter disagreements."""
    import random
    random.seed(42)

    qa_indices = random.sample(range(len(samples)), int(len(samples) * sample_rate))
    qa_samples = [samples[i] for i in qa_indices]

    print(f"\nSonnet QA pass on {len(qa_samples)} samples ({sample_rate*100:.0f}%)...")

    requests = create_batch_requests(qa_samples, model)
    batch_id = submit_batch(client, requests)
    results = wait_for_batch(client, batch_id, model, len(qa_samples))

    batch_cost = tracker.add_batch(model, results, len(qa_samples))
    print(f"Sonnet QA batch cost: ${batch_cost:.4f}")

    disagreements = 0
    qa_result_map = {}
    for r in results:
        if r.result.type != "succeeded":
            continue
        try:
            idx = int(r.custom_id.split("-")[1])
            content = r.result.message.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            qa_result_map[qa_indices[idx]] = parsed
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            continue

    disagreement_indices = set()
    for orig_idx, sonnet_label in qa_result_map.items():
        if samples[orig_idx]["label"] != sonnet_label.get("label"):
            disagreement_indices.add(orig_idx)
            disagreements += 1
        else:
            for cat in CATEGORIES:
                if cat in sonnet_label.get("categories", {}):
                    samples[orig_idx]["categories"][cat] = sonnet_label["categories"][cat]

    agreement_rate = 1 - (disagreements / len(qa_result_map)) if qa_result_map else 0
    print(f"Agreement rate: {agreement_rate*100:.1f}%")
    print(f"Removing {len(disagreement_indices)} disagreements")

    filtered = [s for i, s in enumerate(samples) if i not in disagreement_indices]
    return filtered


def main():
    config = load_config()
    label_config = config["labeling"]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    tracker = BatchCostTracker()

    # Load all raw data
    all_samples = []
    for jsonl_file in sorted(RAW_DIR.glob("*.jsonl")):
        data = load_jsonl(jsonl_file)
        print(f"Loaded {len(data)} from {jsonl_file.name}")
        all_samples.extend(data)

    print(f"\nTotal samples to label: {len(all_samples)}")

    # Step 1: Haiku batch labeling
    print(f"\n{'='*60}")
    print(f"Step 1: Haiku labeling ({label_config['haiku_model']})")
    print(f"Batch API pricing: ${HAIKU_INPUT_PER_TOKEN*1e6:.2f}/MTok in, ${HAIKU_OUTPUT_PER_TOKEN*1e6:.2f}/MTok out")
    print(f"{'='*60}")

    chunk_size = 10_000
    all_labeled = []
    for chunk_start in range(0, len(all_samples), chunk_size):
        chunk_samples = all_samples[chunk_start:chunk_start + chunk_size]
        chunk_num = chunk_start // chunk_size + 1
        total_chunks = (len(all_samples) + chunk_size - 1) // chunk_size
        print(f"\nChunk {chunk_num}/{total_chunks} ({len(chunk_samples)} samples)...")

        # Create requests per-chunk so custom_ids start at 0 for each batch
        chunk_requests = create_batch_requests(chunk_samples, label_config["haiku_model"])
        batch_id = submit_batch(client, chunk_requests)
        results = wait_for_batch(client, batch_id, label_config["haiku_model"], len(chunk_samples))
        batch_cost = tracker.add_batch(label_config["haiku_model"], results, len(chunk_samples))
        print(f"Chunk {chunk_num} cost: ${batch_cost:.4f}")

        labeled = parse_batch_results(results, chunk_samples)
        all_labeled.extend(labeled)
        tracker.report(f"\nRunning total after chunk {chunk_num}:")

    save_jsonl(all_labeled, LABELED_DIR / "haiku_labeled.jsonl")
    print(f"\nHaiku labeled: {len(all_labeled)} samples")
    tracker.report("After Haiku labeling:")

    # Step 2: Sonnet QA pass
    print(f"\n{'='*60}")
    print(f"Step 2: Sonnet QA ({label_config['sonnet_model']})")
    print(f"Batch API pricing: ${SONNET_INPUT_PER_TOKEN*1e6:.2f}/MTok in, ${SONNET_OUTPUT_PER_TOKEN*1e6:.2f}/MTok out")
    print(f"{'='*60}")

    final = run_sonnet_qa(
        client, all_labeled,
        label_config["sonnet_model"],
        label_config["sonnet_qa_sample_rate"],
        tracker,
    )

    save_jsonl(final, LABELED_DIR / "final_labeled.jsonl")

    # Final summary
    total = len(final)
    unsafe = sum(1 for s in final if s["label"] == "unsafe")
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Final labeled dataset: {total} samples")
    print(f"  Safe:   {total - unsafe} ({(total-unsafe)/total*100:.1f}%)")
    print(f"  Unsafe: {unsafe} ({unsafe/total*100:.1f}%)")
    tracker.report("Final cost:")


if __name__ == "__main__":
    main()

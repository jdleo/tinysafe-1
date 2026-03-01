# safety-distill

A 22M parameter safety classifier built on DeBERTa-v3-xsmall. Dual-head architecture: binary safe/unsafe classification + 7-category multi-label breakdown (violence, hate, sexual, self-harm, dangerous info, harassment, illegal activity).

Trained on ~41K samples from public safety datasets (WildGuard, BeaverTails, ToxiGen, ToxicChat, XSTest, HarmBench, SORRY-Bench) plus synthetic data, labeled via Claude Batch API with Sonnet QA verification.

## Results

| Model | Params | WildGuard F1 | ToxicChat F1 | OR-Bench FPR | Latency |
|---|---|---|---|---|---|
| **safety-distill** | **22M** | **0.75** | **0.59** | **18.9%** | **~2ms** |
| WildGuard-7B | 7B | ~0.90 | ~0.92 | ~10% | ~500ms |
| LlamaGuard-3-8B | 8B | ~0.88 | ~0.90 | ~12% | ~600ms |
| ToxicBERT | 110M | ~0.78 | ~0.82 | ~25% | ~10ms |

Internal test set: 0.87 F1 macro, 0.90 unsafe recall, 0.82 unsafe precision.

## Quickstart

```bash
uv sync
uv run python scripts/05_train.py
```

## Pipeline

| Script | What it does |
|---|---|
| `01_download_datasets.py` | Downloads and normalizes public safety datasets |
| `02_generate_synthetic.py` | Generates synthetic training data via Claude |
| `03_label_with_claude.py` | Labels all data through Claude Batch API |
| `04_quality_filter.py` | Dedup, contamination filter, train/val/test split |
| `05_train.py` | Trains the dual-head DeBERTa classifier |
| `05b_leave_one_out.py` | Leave-one-dataset-out robustness eval |
| `05c_threshold_sweep.py` | Sweeps binary decision threshold on val set |
| `06_evaluate.py` | Benchmarks against ToxicChat, WildGuardBench, OR-Bench |
| `07_export.py` | Exports to ONNX (fp32 + INT8 quantized) |

## Architecture

```
DeBERTa-v3-xsmall (384-dim)
    |
  [CLS] token
   / \
  /   \
Binary  Category
Head    Head (7)
```

Binary head uses focal loss (gamma=2.0). Category head uses BCE weighted at 0.5x. AdamW with linear warmup + decay.

## Config

All hyperparameters live in `configs/config.json`. Key settings:

- Batch size: 32 (effective 64 with grad accumulation)
- LR: 2e-5, weight decay: 0.01
- Early stopping on unsafe recall, patience 2
- Binary threshold: 0.45

## License

MIT

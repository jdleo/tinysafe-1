# TinySafe v1

71M parameter safety classifier built on DeBERTa-v3-xsmall. Dual-head architecture: binary safe/unsafe + 7-category multi-label (violence, hate, sexual, self-harm, dangerous info, harassment, illegal activity).

Trained on ~41K samples from public safety datasets (WildGuard, BeaverTails, ToxiGen, ToxicChat, XSTest, HarmBench, SORRY-Bench) plus synthetic data, labeled via Claude Batch API with Sonnet QA verification.

**Model on HuggingFace:** [jdleo/tinysafe-1](https://huggingface.co/jdleo/tinysafe-1)

## ToxicChat F1

| Model | Params | F1 |
|---|---|---|
| Qwen3Guard-8B | 8B | 73% |
| AprielGuard-8B | 8B | 72% |
| Granite Guardian-8B | 8B | 71% |
| Granite Guardian-3B | 3B | 68% |
| ShieldGemma-2B | 2B | 67% |
| Qwen3Guard-0.6B | 0.6B | 63% |
| **TinySafe v1** | **71M** | **59%** |
| LlamaGuard 3-8B | 8B | 51% |
| ShieldGemma-27B | 27B | 48% |
| LlamaGuard 4-12B | 12B | 45% |
| ToxDectRoberta | 125M | 34.6% |
| HateBERT | 110M | 11.6% |

Beats LlamaGuard 3 (8B), ShieldGemma (27B), LlamaGuard 4 (12B), and all encoder-based models in its class. 100x smaller than the nearest guard model that outperforms it.

## WildGuardBench

| Model | Params | WGTest F1 |
|---|---|---|
| GPT-4 | — | 77.3 |
| WildGuard | 7B | 75.4 |
| **TinySafe v1** | **71M** | **75.0** |
| LlamaGuard 2 | 8B | 66.5 |
| BeaverDam | 7B | 63.4 |
| LlamaGuard | 7B | 50.5 |

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

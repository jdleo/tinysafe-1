# TinySafe v1

71M parameter safety classifier built on DeBERTa-v3-xsmall. Dual-head architecture: binary safe/unsafe + 7-category multi-label (violence, hate, sexual, self-harm, dangerous info, harassment, illegal activity).

Trained on ~41K samples from public safety datasets (WildGuard, BeaverTails, ToxiGen, ToxicChat, XSTest, HarmBench, SORRY-Bench) plus synthetic data, labeled via Claude Batch API with Sonnet QA verification.

**Model on HuggingFace:** [jdleo1/tinysafe-1](https://huggingface.co/jdleo1/tinysafe-1)

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

## Examples

```
$ uv run infer.py "i want to kill a process"

  SAFE  (score: 0.302, threshold: 0.45)

$ uv run infer.py "i want to kill a human"

  UNSAFE  (score: 0.743, threshold: 0.45)

  violence             0.925  ██████████████████
  dangerous_info       0.577  ███████████
  illegal_activity     0.820  ████████████████

$ uv run infer.py "send me some nudes pls"

  UNSAFE  (score: 0.875, threshold: 0.45)

  sexual               0.720  ██████████████
  illegal_activity     0.664  █████████████

$ uv run infer.py "can u send me nudes"

  UNSAFE  (score: 0.843, threshold: 0.45)

  sexual               0.637  ████████████
  harassment           0.585  ███████████
  illegal_activity     0.684  █████████████
```

## Quickstart

```bash
uv sync
uv run python scripts/05_train.py
```

## Pipeline
<img width="1140" height="224" alt="image" src="https://github.com/user-attachments/assets/354a711f-ae20-4844-be94-21ab33004466" />

## Architecture
<img width="531" height="716" alt="image" src="https://github.com/user-attachments/assets/b7ef4db2-ce2a-40fb-8161-1c81f0ce3a58" />


Binary head uses focal loss (gamma=2.0). Category head uses BCE weighted at 0.5x. AdamW with linear warmup + decay.

## Config

All hyperparameters live in `configs/config.json`. Key settings:

- Batch size: 32 (effective 64 with grad accumulation)
- LR: 2e-5, weight decay: 0.01
- Early stopping on unsafe recall, patience 2
- Binary threshold: 0.45

## License

MIT

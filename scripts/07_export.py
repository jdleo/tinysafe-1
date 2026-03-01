#!/usr/bin/env python3
"""Export trained model to ONNX + INT8 quantization."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import DebertaV2Tokenizer

from src.model import SafetyClassifier
from src.utils import CATEGORIES, load_config

CHECKPOINT_DIR = Path("checkpoints")
EXPORT_DIR = Path("exported")


def export_onnx(model, tokenizer, config, export_path: Path):
    """Export to ONNX format."""
    model.eval()
    model = model.to("cpu")

    dummy_text = "This is a sample text for export."
    inputs = tokenizer(
        dummy_text,
        truncation=True,
        max_length=config["max_length"],
        padding="max_length",
        return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(export_path),
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["binary_logit", "category_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "binary_logit": {0: "batch"},
            "category_logits": {0: "batch"},
        },
    )

    size_mb = export_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX fp32: {export_path} ({size_mb:.1f} MB)")
    return export_path


def quantize_int8(onnx_path: Path, quantized_path: Path):
    """Quantize ONNX model to INT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    size_mb = quantized_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX INT8: {quantized_path} ({size_mb:.1f} MB)")
    return quantized_path


def benchmark_inference(tokenizer, config):
    """Benchmark inference latency for all formats."""
    import onnxruntime as ort

    test_texts = [
        "Hello, how are you doing today?",
        "I want to hurt someone badly",
        "Can you explain the history of World War II?",
        "How do I make a pipe bomb?",
        "The doctor prescribed medication for my anxiety",
    ]

    print("\nInference Benchmark (5 samples, avg of 100 runs):")

    # ONNX fp32
    onnx_path = EXPORT_DIR / "model.onnx"
    if onnx_path.exists():
        session = ort.InferenceSession(str(onnx_path))
        times = []
        for _ in range(100):
            for text in test_texts:
                inputs = tokenizer(text, truncation=True, max_length=config["max_length"],
                                   padding="max_length", return_tensors="np")
                start = time.perf_counter()
                session.run(None, {"input_ids": inputs["input_ids"],
                                   "attention_mask": inputs["attention_mask"]})
                times.append(time.perf_counter() - start)
        avg_ms = np.mean(times) * 1000
        print(f"  ONNX fp32: {avg_ms:.2f} ms/sample")

    # ONNX INT8
    int8_path = EXPORT_DIR / "model_int8.onnx"
    if int8_path.exists():
        session = ort.InferenceSession(str(int8_path))
        times = []
        for _ in range(100):
            for text in test_texts:
                inputs = tokenizer(text, truncation=True, max_length=config["max_length"],
                                   padding="max_length", return_tensors="np")
                start = time.perf_counter()
                session.run(None, {"input_ids": inputs["input_ids"],
                                   "attention_mask": inputs["attention_mask"]})
                times.append(time.perf_counter() - start)
        avg_ms = np.mean(times) * 1000
        print(f"  ONNX INT8: {avg_ms:.2f} ms/sample")


def create_inference_module(export_dir: Path):
    """Create a standalone inference script."""
    code = '''#!/usr/bin/env python3
"""Standalone inference for the safety classifier."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import DebertaV2Tokenizer

CATEGORIES = [
    "violence", "hate", "sexual", "self_harm",
    "dangerous_info", "harassment", "illegal_activity",
]

MODEL_DIR = Path(__file__).parent


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SafetyClassifierInference:
    def __init__(self, model_path: str = None, tokenizer_name: str = "microsoft/deberta-v3-xsmall"):
        if model_path is None:
            model_path = str(MODEL_DIR / "model_int8.onnx")
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_name, use_fast=False)

    def classify(self, text: str, threshold: float = 0.5) -> dict:
        inputs = self.tokenizer(
            text, truncation=True, max_length=512,
            padding="max_length", return_tensors="np",
        )
        binary_logit, category_logits = self.session.run(
            None,
            {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
        )

        unsafe_score = sigmoid(binary_logit[0][0])
        category_scores = sigmoid(category_logits[0])

        return {
            "label": "unsafe" if unsafe_score > threshold else "safe",
            "confidence": float(max(unsafe_score, 1 - unsafe_score)),
            "unsafe_score": float(unsafe_score),
            "categories": dict(zip(CATEGORIES, [float(s) for s in category_scores])),
        }


if __name__ == "__main__":
    import sys
    classifier = SafetyClassifierInference()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to classify: ")
    result = classifier.classify(text)
    print(json.dumps(result, indent=2))
'''
    inference_path = export_dir / "inference.py"
    with open(inference_path, "w") as f:
        f.write(code)
    print(f"  Inference module: {inference_path}")


def main():
    config = load_config()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"], use_fast=False)

    # Load best model
    print("Loading best model...")
    model = SafetyClassifier(config["base_model"], config["num_categories"])
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Export ONNX
    print("\nExporting ONNX...")
    onnx_path = export_onnx(model, tokenizer, config, EXPORT_DIR / "model.onnx")

    # Quantize INT8
    print("\nQuantizing to INT8...")
    int8_path = quantize_int8(onnx_path, EXPORT_DIR / "model_int8.onnx")

    # PyTorch checkpoint size
    pt_size = (CHECKPOINT_DIR / "best_model.pt").stat().st_size / (1024 * 1024)
    print(f"\n  PyTorch fp32: {pt_size:.1f} MB")

    # Create inference module
    print("\nCreating inference module...")
    create_inference_module(EXPORT_DIR)

    # Save tokenizer alongside model
    tokenizer.save_pretrained(EXPORT_DIR / "tokenizer")
    print(f"  Tokenizer saved to {EXPORT_DIR / 'tokenizer'}")

    # Benchmark
    benchmark_inference(tokenizer, config)

    # Summary
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"  ONNX fp32: {EXPORT_DIR / 'model.onnx'}")
    print(f"  ONNX INT8: {EXPORT_DIR / 'model_int8.onnx'}")
    print(f"  Inference: {EXPORT_DIR / 'inference.py'}")
    print(f"  Tokenizer: {EXPORT_DIR / 'tokenizer'}")


if __name__ == "__main__":
    main()

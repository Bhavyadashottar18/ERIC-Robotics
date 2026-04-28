"""
scripts/evaluate.py
───────────────────
Evaluate a trained model and optionally apply + compare optimizations.

    # Evaluate only
    python scripts/evaluate.py --weights runs/train/weights/best.pt

    # Evaluate + quantize + benchmark
    python scripts/evaluate.py --weights runs/train/weights/best.pt --optimize all

    # Only quantize
    python scripts/evaluate.py --weights runs/train/weights/best.pt --optimize quant

    # Only prune
    python scripts/evaluate.py --weights runs/train/weights/best.pt --optimize prune
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset  import BDD100KNavDataset
from src.model    import NavDetector
from src.optimize import ModelOptimizer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and optimize the navigation model")
    p.add_argument("--weights", required=True, help="Trained model weights (.pt)")
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument(
        "--optimize",
        choices=["none", "quant", "prune", "all"],
        default="none",
        help="Optimization to apply after evaluation",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Validate ──────────────────────────────────────────────────────────────
    dataset   = BDD100KNavDataset(cfg)
    yaml_path = dataset.write_yaml()

    detector  = NavDetector(cfg)
    detector.load_weights(args.weights)

    print("\n" + "=" * 50)
    print("Baseline Model Evaluation")
    print("=" * 50)
    metrics = detector.validate(yaml_path)
    print(f"  mAP@50      : {metrics['mAP50']:.4f}")
    print(f"  mAP@50:95   : {metrics['mAP50_95']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")

    if args.optimize == "none":
        return

    opt = ModelOptimizer(cfg, weights_path=args.weights)

    # ── Quantization ──────────────────────────────────────────────────────────
    if args.optimize in ("quant", "all"):
        print("\n" + "=" * 50)
        print("Quantization (INT8)")
        print("=" * 50)
        try:
            int8_path = opt.quantize_onnx()
            print(f"  INT8 model saved: {int8_path}")
        except Exception as e:
            print(f"  Quantization failed: {e}")

    # ── Pruning ───────────────────────────────────────────────────────────────
    if args.optimize in ("prune", "all"):
        print("\n" + "=" * 50)
        print("L1 Unstructured Pruning (30 %)")
        print("=" * 50)
        try:
            pruned_path = opt.prune_pytorch()
            print(f"  Pruned model saved: {pruned_path}")
        except Exception as e:
            print(f"  Pruning failed: {e}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if args.optimize == "all":
        print("\n" + "=" * 50)
        print("FPS Benchmark (CPU vs GPU)")
        print("=" * 50)
        opt.benchmark_all()


if __name__ == "__main__":
    main()

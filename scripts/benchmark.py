"""
scripts/benchmark.py
────────────────────
Quick standalone FPS benchmark across model variants and devices.

    python scripts/benchmark.py --weights runs/train/weights/best.pt
    python scripts/benchmark.py --weights runs/train/weights/best.pt --n-runs 200
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimize import ModelOptimizer


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark model FPS on CPU and GPU")
    p.add_argument("--weights", required=True, help="Base .pt weights")
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--n-runs",  type=int, default=100, help="Inference runs for timing")
    p.add_argument("--imgsz",   type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    opt = ModelOptimizer(cfg, weights_path=args.weights)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          Navigation Model — FPS Benchmark                ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}\n")
    else:
        print("No GPU detected — CPU only.\n")

    # Baseline PT
    for dev in devices:
        m = opt.benchmark_fps(
            args.weights, n_runs=args.n_runs, imgsz=args.imgsz, device=dev
        )
        print(f"[Baseline YOLOv8n .pt] {dev.upper():<4}  "
              f"FPS: {m['fps']:6.1f}  |  avg: {m['avg_ms']:.1f} ms")

    # ONNX FP32
    onnx_fp32 = str(Path(cfg["model"]["output_dir"]) / "model_fp32.onnx")
    onnx_int8 = str(Path(cfg["model"]["output_dir"]) / "model_int8.onnx")

    for label, path in [("ONNX FP32", onnx_fp32), ("ONNX INT8", onnx_int8)]:
        if not Path(path).exists():
            print(f"[{label}]  Not found — run evaluate.py --optimize quant first.")
            continue
        m = opt.benchmark_fps(path, n_runs=args.n_runs, imgsz=args.imgsz, device="cpu")
        print(f"[{label}]          CPU   "
              f"FPS: {m['fps']:6.1f}  |  avg: {m['avg_ms']:.1f} ms")

    print("\nDone.")


if __name__ == "__main__":
    main()

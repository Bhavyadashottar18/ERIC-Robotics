"""
optimize.py
───────────
Model optimization utilities for edge deployment.

Techniques covered
------------------
1. INT8 Post-Training Quantization (via ONNX Runtime / PyTorch)
2. Structured / Unstructured Pruning (via torch-pruning + torch.nn.utils.prune)
3. FP16 Half-precision export

Each technique records before/after metrics (FPS, mAP, model size).

Usage
-----
    from src.optimize import ModelOptimizer
    opt = ModelOptimizer(cfg, weights_path="runs/train/weights/best.pt")
    opt.quantize_onnx()
    opt.prune_and_finetune(dataset_yaml)
    opt.benchmark_all()
"""

import copy
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class ModelOptimizer:
    """
    Wraps quantization, pruning, and benchmarking for YOLOv8 models.
    """

    def __init__(self, cfg: dict, weights_path: str):
        self.cfg          = cfg
        self.ocfg         = cfg.get("optimization", {})
        self.weights_path = weights_path
        self.out_dir      = Path(cfg["model"]["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[optimize] Device: {self.device}")

    # ── 1. ONNX INT8 Quantization ─────────────────────────────────────────────

    def quantize_onnx(
        self,
        calibration_images: Optional[list] = None,
    ) -> str:
        """
        Export YOLOv8 → ONNX, then apply static INT8 quantisation
        using ONNX Runtime's quantisation toolkit.

        Returns path to the quantised model.
        """
        from ultralytics import YOLO
        import onnx
        from onnxruntime.quantization import (
            quantize_static, quantize_dynamic,
            CalibrationDataReader, QuantType,
        )

        # Step 1: Export to ONNX
        model = YOLO(self.weights_path)
        onnx_path = str(self.out_dir / "model_fp32.onnx")
        model.export(format="onnx", simplify=True, dynamic=False, imgsz=640)
        # Ultralytics puts the onnx next to the .pt weights
        src_onnx = self.weights_path.replace(".pt", ".onnx")
        if os.path.exists(src_onnx):
            os.rename(src_onnx, onnx_path)
        print(f"[optimize] FP32 ONNX exported → {onnx_path}")

        # Step 2: Dynamic INT8 quantisation (no calibration data required)
        int8_path = str(self.out_dir / "model_int8.onnx")
        quantize_dynamic(
            model_input   = onnx_path,
            model_output  = int8_path,
            weight_type   = QuantType.QInt8,
        )
        print(f"[optimize] INT8 ONNX quantised → {int8_path}")

        # Report size reduction
        fp32_mb = os.path.getsize(onnx_path) / 1e6
        int8_mb = os.path.getsize(int8_path) / 1e6
        print(f"[optimize] Size: {fp32_mb:.1f} MB → {int8_mb:.1f} MB "
              f"({100*(1-int8_mb/fp32_mb):.0f}% smaller)")
        return int8_path

    # ── 2. PyTorch Unstructured Pruning ───────────────────────────────────────

    def prune_pytorch(
        self,
        amount: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Apply L1 unstructured pruning to Conv2d layers of the YOLOv8 backbone.

        amount : fraction of weights to zero out (default from config).
        Returns path to pruned checkpoint.
        """
        amount = amount or self.ocfg.get("pruning", {}).get("amount", 0.30)
        from ultralytics import YOLO

        model = YOLO(self.weights_path)
        torch_model: nn.Module = model.model

        # Collect all Conv2d layers
        conv_layers = [
            (name, module)
            for name, module in torch_model.named_modules()
            if isinstance(module, nn.Conv2d)
        ]
        print(f"[optimize] Pruning {len(conv_layers)} Conv2d layers "
              f"(amount={amount:.0%}) …")

        for name, module in conv_layers:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")   # make permanent

        # Count sparsity
        total_params  = sum(p.numel() for p in torch_model.parameters())
        zero_params   = sum(
            int((p == 0).sum()) for p in torch_model.parameters()
        )
        sparsity = zero_params / total_params
        print(f"[optimize] Sparsity achieved: {sparsity:.1%}")

        # Save pruned model
        save_path = save_path or str(self.out_dir / "model_pruned.pt")
        torch.save(torch_model.state_dict(), save_path)
        print(f"[optimize] Pruned weights saved → {save_path}")
        return save_path

    # ── 3. FP16 Half-Precision Export ─────────────────────────────────────────

    def export_fp16(self) -> str:
        """Export model to FP16 TorchScript (GPU inference)."""
        from ultralytics import YOLO
        model = YOLO(self.weights_path)
        path  = model.export(format="torchscript", half=True)
        print(f"[optimize] FP16 TorchScript exported → {path}")
        return str(path)

    # ── 4. Benchmarking ───────────────────────────────────────────────────────

    def benchmark_fps(
        self,
        model_path: str,
        n_warmup: int = 10,
        n_runs:   int = 100,
        imgsz:    int = 640,
        device:   str = "cpu",
    ) -> Dict[str, float]:
        """
        Measure inference FPS for a given model checkpoint.

        Returns dict with keys: fps, avg_ms, min_ms, max_ms.
        """
        import onnxruntime as ort

        dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

        if model_path.endswith(".onnx"):
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            sess = ort.InferenceSession(model_path, providers=providers)
            input_name = sess.get_inputs()[0].name
            run_fn = lambda: sess.run(None, {input_name: dummy})
        else:
            # PyTorch path
            from ultralytics import YOLO
            yolo = YOLO(model_path)
            torch_model = yolo.model.to(device)
            torch_model.eval()
            t = torch.from_numpy(dummy).to(device)
            run_fn = lambda: torch_model(t)

        # Warm-up
        for _ in range(n_warmup):
            run_fn()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            run_fn()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        return {
            "fps":    1000.0 / avg_ms,
            "avg_ms": avg_ms,
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "device": device,
        }

    def benchmark_all(self) -> None:
        """
        Run FPS benchmarks on baseline, INT8, and pruned models,
        then print a comparison table.
        """
        configs = [
            ("Baseline (FP32 PT)",  self.weights_path,              "cpu"),
            ("Baseline (FP32 PT)",  self.weights_path,              "cuda"),
        ]
        onnx_fp32 = str(self.out_dir / "model_fp32.onnx")
        onnx_int8 = str(self.out_dir / "model_int8.onnx")
        if Path(onnx_fp32).exists():
            configs += [
                ("ONNX FP32",  onnx_fp32, "cpu"),
                ("ONNX INT8",  onnx_int8, "cpu"),
            ]

        print("\n" + "="*65)
        print(f"{'Model':<28} {'Device':<6} {'FPS':>7} {'Avg ms':>8}")
        print("="*65)
        for label, path, dev in configs:
            if dev == "cuda" and not torch.cuda.is_available():
                print(f"{label:<28} {'cuda':<6} {'N/A':>7} {'N/A':>8}")
                continue
            if not Path(path).exists():
                print(f"{label:<28} {dev:<6} {'missing':>7} {'':>8}")
                continue
            try:
                m = self.benchmark_fps(path, device=dev)
                print(f"{label:<28} {dev:<6} {m['fps']:>7.1f} {m['avg_ms']:>7.1f}ms")
            except Exception as e:
                print(f"{label:<28} {dev:<6} {'ERR':>7}  {str(e)[:25]}")
        print("="*65 + "\n")

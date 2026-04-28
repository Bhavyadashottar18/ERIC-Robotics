"""
model.py
────────
YOLOv8 model wrapper for navigation object detection.

Handles:
  • Loading pretrained YOLOv8 weights
  • Fine-tuning on the navigation subset of BDD100K
  • Saving / loading checkpoints
  • Raw inference returning structured results

Usage
-----
    from src.model import NavDetector
    detector = NavDetector(cfg)
    detector.train(dataset_yaml)
    results  = detector.predict("image.jpg")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from ultralytics import YOLO


CLASS_NAMES = ["traffic cone", "barrier", "stop sign"]


class NavDetector:
    """
    Thin wrapper around Ultralytics YOLOv8 tuned for the three
    navigation classes: traffic cone, barrier, stop sign.
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.mcfg     = cfg["model"]
        self.tcfg     = cfg["training"]
        self.icfg     = cfg["inference"]
        self.out_dir  = Path(self.mcfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        weights = self.mcfg["weights"]
        print(f"[model] Loading weights: {weights}")
        self.model = YOLO(weights)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, dataset_yaml: str) -> str:
        """
        Fine-tune on the navigation dataset.

        Returns path to the best checkpoint.
        """
        print(f"[model] Starting fine-tuning — {self.tcfg['epochs']} epochs …")

        results = self.model.train(
            data        = dataset_yaml,
            epochs      = self.tcfg["epochs"],
            batch       = self.tcfg["batch_size"],
            imgsz       = self.tcfg["imgsz"],
            lr0         = self.tcfg["lr0"],
            lrf         = self.tcfg["lrf"],
            momentum    = self.tcfg["momentum"],
            weight_decay= self.tcfg["weight_decay"],
            warmup_epochs=self.tcfg["warmup_epochs"],
            patience    = self.tcfg["patience"],
            device      = self.tcfg["device"] or ("0" if torch.cuda.is_available() else "cpu"),
            workers     = self.tcfg["workers"],
            augment     = self.tcfg["augment"],
            cache       = self.tcfg["cache"],
            project     = str(self.out_dir),
            name        = "train",
            exist_ok    = True,
            verbose     = True,
        )

        best = str(Path(results.save_dir) / "weights" / "best.pt")
        print(f"[model] Training complete. Best checkpoint → {best}")
        return best

    # ── Inference ─────────────────────────────────────────────────────────────

    def load_weights(self, weights_path: str) -> None:
        """Load a fine-tuned checkpoint for inference."""
        print(f"[model] Loading fine-tuned weights: {weights_path}")
        self.model = YOLO(weights_path)

    def predict(
        self,
        source: Any,
        conf: Optional[float] = None,
        iou:  Optional[float] = None,
    ) -> List[Dict]:
        """
        Run inference on a source (path, np.ndarray, or list).

        Returns a list of dicts, one per image:
            {
              "boxes":   [[x1,y1,x2,y2], ...],
              "scores":  [float, ...],
              "classes": [int, ...],
              "names":   [str, ...],
            }
        """
        conf = conf or self.icfg["conf_threshold"]
        iou  = iou  or self.icfg["iou_threshold"]

        raw = self.model.predict(
            source    = source,
            conf      = conf,
            iou       = iou,
            max_det   = self.icfg["max_det"],
            verbose   = False,
        )

        parsed: List[Dict] = []
        for r in raw:
            boxes   = r.boxes.xyxy.cpu().numpy().tolist()    # [[x1,y1,x2,y2]]
            scores  = r.boxes.conf.cpu().numpy().tolist()
            classes = r.boxes.cls.cpu().numpy().astype(int).tolist()
            names   = [self.model.names.get(c, "unknown") for c in classes]
            parsed.append({
                "boxes":   boxes,
                "scores":  scores,
                "classes": classes,
                "names":   names,
                "orig_img": r.orig_img,
            })
        return parsed

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, dataset_yaml: str) -> Dict:
        """Run YOLO validation and return metrics dict."""
        metrics = self.model.val(data=dataset_yaml, verbose=True)
        return {
            "mAP50":   metrics.box.map50,
            "mAP50_95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall":    metrics.box.mr,
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def export_onnx(self, output_path: Optional[str] = None) -> str:
        """Export model to ONNX format."""
        path = self.model.export(format="onnx", dynamic=True, simplify=True)
        if output_path:
            import shutil
            shutil.move(str(path), output_path)
            return output_path
        return str(path)

    def export_torchscript(self) -> str:
        """Export to TorchScript for CPU-optimised deployment."""
        path = self.model.export(format="torchscript")
        return str(path)

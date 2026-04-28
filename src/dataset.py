"""
dataset.py
──────────
BDD100K dataset loader and YOLO-format converter.

Matches the ACTUAL downloaded BDD100K folder structure:

  bdd100k/
    bdd100k/
      images/
        100k/
          train/   *.jpg
          val/     *.jpg

  bdd100k_labels_release/
    bdd100k/
      labels/
        bdd100k_labels_images_train.json
        bdd100k_labels_images_val.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml
from tqdm import tqdm


TARGET_CATEGORIES: Dict[str, int] = {
    "traffic cone": 0,
    "barrier":      1,
    "stop sign":    2,
    "traffic sign": 2,
}

CLASS_NAMES = ["traffic cone", "barrier", "stop sign"]


class BDD100KNavDataset:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        ds = cfg["dataset"]

        self.images = {
            "train": Path(ds["images_train"]),
            "val":   Path(ds["images_val"]),
        }
        self.json_labels = {
            "train": Path(ds["labels_json_train"]),
            "val":   Path(ds["labels_json_val"]),
        }
        self.yolo_labels = {
            "train": Path(ds["labels_train"]),
            "val":   Path(ds["labels_val"]),
        }
        self.yaml_path = Path(ds["yaml_path"])

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self) -> None:
        """Convert JSON labels to YOLO .txt files for train and val splits."""
        for split in ("train", "val"):
            json_path = self.json_labels[split]
            if not json_path.exists():
                print(f"[dataset] WARNING: {json_path} not found — skipping {split}.")
                print(f"          Check the path in config.yaml")
                continue
            self._convert_split(json_path, split)
        print("[dataset] Preparation complete.")

    def write_yaml(self) -> str:
        """Write the YOLO dataset YAML and return its path."""
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "path":  "/",
            "train": str(self.images["train"].resolve()),
            "val":   str(self.images["val"].resolve()),
            "nc":    len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"[dataset] Dataset YAML written to {self.yaml_path}")
        return str(self.yaml_path)

    def class_distribution(self, split: str = "train") -> Dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        lbl_dir = self.yolo_labels[split]
        if not lbl_dir.exists():
            return counts
        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.split()
                if parts:
                    cls_idx = int(parts[0])
                    if cls_idx < len(CLASS_NAMES):
                        counts[CLASS_NAMES[cls_idx]] += 1
        return counts

    # ── Internals ─────────────────────────────────────────────────────────────

    def _convert_split(self, json_path: Path, split: str) -> None:
        out_dir = self.yolo_labels[split]
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[dataset] Reading {json_path.name} ({json_path.stat().st_size/1e6:.0f} MB) ...")
        with open(json_path) as f:
            data: List[dict] = json.load(f)

        kept, skipped = 0, 0
        for entry in tqdm(data, desc=f"[dataset] Converting {split}"):
            labels    = entry.get("labels") or []
            img_name  = entry.get("name", "")
            yolo_lines = self._labels_to_yolo(labels, img_name, split)
            if not yolo_lines:
                skipped += 1
                continue
            stem = Path(img_name).stem
            (out_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))
            kept += 1

        print(f"[dataset]  {split}: {kept} images kept, {skipped} skipped")

    def _labels_to_yolo(
        self, labels: List[dict], img_name: str, split: str
    ) -> List[str]:
        lines: List[str] = []
        img_path = self.images[split] / img_name
        w, h = self._get_image_size(img_path)

        for lbl in labels:
            category = lbl.get("category", "").lower()
            cls_idx  = TARGET_CATEGORIES.get(category)
            if cls_idx is None:
                continue
            box2d = lbl.get("box2d")
            if box2d is None:
                continue
            x1, y1, x2, y2 = (
                box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            )
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cx, cy, bw, bh = (max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh))
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    @staticmethod
    def _get_image_size(img_path: Path) -> Tuple[int, int]:
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                return img.shape[1], img.shape[0]
        return 1280, 720

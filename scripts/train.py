"""
scripts/train.py
────────────────
Fine-tune YOLOv8 on the BDD100K navigation subset.

    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --epochs 100 --batch 32
"""

import argparse
import sys
from pathlib import Path

import yaml

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import BDD100KNavDataset
from src.model   import NavDetector


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on BDD100K Nav subset")
    p.add_argument("--config",  default="configs/config.yaml", help="Path to config YAML")
    p.add_argument("--epochs",  type=int,   default=None, help="Override training epochs")
    p.add_argument("--batch",   type=int,   default=None, help="Override batch size")
    p.add_argument("--weights", default=None, help="Override starting weights")
    p.add_argument("--skip-prep", action="store_true",
                   help="Skip dataset conversion (labels already prepared)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs:  cfg["training"]["epochs"]     = args.epochs
    if args.batch:   cfg["training"]["batch_size"] = args.batch
    if args.weights: cfg["model"]["weights"]        = args.weights

    # ── Step 1: Prepare dataset ───────────────────────────────────────────────
    dataset = BDD100KNavDataset(cfg)
    if not args.skip_prep:
        print("=" * 60)
        print("STEP 1/2 — Preparing BDD100K navigation dataset …")
        print("=" * 60)
        dataset.prepare()
    yaml_path = dataset.write_yaml()

    # Print class distribution
    for split in ("train", "val"):
        dist = dataset.class_distribution(split)
        print(f"[train] {split} class distribution: {dist}")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2/2 — Fine-tuning YOLOv8 …")
    print("=" * 60)
    detector  = NavDetector(cfg)
    best_ckpt = detector.train(yaml_path)

    # ── Step 3: Quick validation ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Validating best checkpoint …")
    print("=" * 60)
    detector.load_weights(best_ckpt)
    metrics = detector.validate(yaml_path)
    print(f"\n  mAP@50      : {metrics['mAP50']:.4f}")
    print(f"  mAP@50:95   : {metrics['mAP50_95']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"\n  Best weights: {best_ckpt}")


if __name__ == "__main__":
    main()

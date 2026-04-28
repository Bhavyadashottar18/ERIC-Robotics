"""
scripts/demo.py
───────────────
Run the full detection + distance pipeline on an image or video.

    # Image
    python scripts/demo.py --source data/sample.jpg --weights runs/train/weights/best.pt

    # Video
    python scripts/demo.py --source data/drive.mp4  --weights runs/train/weights/best.pt --show

    # Use pretrained (no fine-tuning) — quick test
    python scripts/demo.py --source data/sample.jpg --weights yolov8n.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import InferencePipeline


def parse_args():
    p = argparse.ArgumentParser(description="Detect + estimate distance on image/video")
    p.add_argument("--source",  required=True,  help="Input image or video path")
    p.add_argument("--weights", required=True,  help="Model weights (.pt)")
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--out",     default=None,   help="Output path (auto if omitted)")
    p.add_argument("--show",    action="store_true", help="Display result window")
    p.add_argument("--conf",    type=float, default=None, help="Confidence threshold")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit frames processed (video only)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.conf:
        cfg["inference"]["conf_threshold"] = args.conf

    pipeline = InferencePipeline(cfg, weights=args.weights)
    source   = args.source
    ext      = Path(source).suffix.lower()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if ext in video_exts:
        pipeline.run_video(
            source,
            out_path   = args.out,
            show       = args.show,
            max_frames = args.max_frames,
        )
    elif ext in image_exts:
        pipeline.run_image(source, out_path=args.out, show=args.show)
    else:
        print(f"[demo] Unknown file type: {ext}. "
              f"Supported: {image_exts | video_exts}")
        sys.exit(1)


if __name__ == "__main__":
    main()

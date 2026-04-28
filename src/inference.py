"""
inference.py
────────────
End-to-end inference pipeline:
    image / video → detect → estimate distance → annotate → save

Usage
-----
    from src.inference import InferencePipeline
    pipeline = InferencePipeline(cfg, weights="runs/train/weights/best.pt")
    pipeline.run_image("frame.jpg", out_path="results/frame_out.jpg")
    pipeline.run_video("drive.mp4", out_path="results/drive_out.mp4")
"""

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.annotator import Annotator
from src.distance  import MonoDistanceEstimator
from src.model     import NavDetector


class InferencePipeline:
    """
    Orchestrates detection, distance estimation, and annotation.
    """

    def __init__(self, cfg: dict, weights: str):
        self.cfg       = cfg
        self.out_dir   = Path(cfg["inference"]["results_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Sub-modules
        self.detector  = NavDetector(cfg)
        self.detector.load_weights(weights)
        self.estimator = MonoDistanceEstimator(cfg)
        self.annotator = Annotator(cfg)

        self.use_flow  = cfg.get("extras", {}).get("optical_flow", False)
        self.use_bev   = cfg.get("extras", {}).get("bird_eye_view", False)

    # ── Single image ──────────────────────────────────────────────────────────

    def run_image(
        self,
        source:   str,
        out_path: Optional[str] = None,
        show:     bool = False,
    ) -> np.ndarray:
        """
        Detect objects, estimate distances, annotate and optionally save.

        Returns the annotated frame (BGR numpy array).
        """
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Cannot load image: {source}")

        annotated = self._process_frame(frame)

        if out_path is None:
            stem     = Path(source).stem
            out_path = str(self.out_dir / f"{stem}_detected.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[inference] Saved → {out_path}")

        if show:
            cv2.imshow("Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated

    # ── Video ─────────────────────────────────────────────────────────────────

    def run_video(
        self,
        source:   str,
        out_path: Optional[str] = None,
        show:     bool = False,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Process each frame of a video. Writes annotated video to out_path.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[inference] Video: {w}×{h} @ {fps:.1f} FPS, {total} frames")

        if out_path is None:
            stem     = Path(source).stem
            out_path = str(self.out_dir / f"{stem}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        prev_gray: Optional[np.ndarray] = None
        frame_idx  = 0
        t_start    = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            annotated  = self._process_frame(frame, prev_gray=prev_gray)
            prev_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            writer.write(annotated)

            if show:
                cv2.imshow("Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"[inference]  {frame_idx}/{total} frames "
                      f"({frame_idx/elapsed:.1f} FPS)")

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        elapsed = time.perf_counter() - t_start
        print(f"[inference] Done. {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx/elapsed:.1f} FPS) → {out_path}")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _process_frame(
        self,
        frame:     np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Full pipeline for one frame: detect → distance → annotate."""
        results   = self.detector.predict(frame)
        if not results:
            return frame

        r         = results[0]
        boxes     = r["boxes"]
        names     = r["names"]
        scores    = r["scores"]
        distances = self.estimator.estimate_batch(boxes, names)

        # Base annotation
        annotated = self.annotator.draw(frame, boxes, names, scores, distances)

        # Optional optical flow
        if self.use_flow and prev_gray is not None:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            annotated = self.annotator.draw_optical_flow(
                annotated, prev_gray, curr_gray, boxes
            )

        # Optional bird's-eye view inset (top-right corner)
        if self.use_bev:
            bev   = self.annotator.bird_eye_view(frame)
            h, w  = frame.shape[:2]
            bev_s = cv2.resize(bev, (w // 3, h // 3))
            annotated[10 : 10 + bev_s.shape[0],
                      w - 10 - bev_s.shape[1] : w - 10] = bev_s
            cv2.rectangle(
                annotated,
                (w - 10 - bev_s.shape[1], 10),
                (w - 10, 10 + bev_s.shape[0]),
                (200, 200, 200), 1,
            )

        return annotated

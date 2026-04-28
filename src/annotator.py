"""
annotator.py
────────────
Draw bounding boxes with class labels and distance annotations on frames.

Each detection is rendered as:
    ┌──────────────────┐
    │  Cone, 1.5m      │
    └──────────────────┘

Also provides bird's-eye view (BEV) transform as an extra-credit feature.

Usage
-----
    from src.annotator import Annotator
    ann = Annotator(cfg)
    frame = ann.draw(frame, boxes, names, scores, distances)
    bev   = ann.bird_eye_view(frame)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Per-class colours (BGR) ───────────────────────────────────────────────────
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "traffic cone": (0,   165, 255),   # orange
    "barrier":      (0,   255, 255),   # yellow
    "stop sign":    (0,   0,   255),   # red
    "unknown":      (128, 128, 128),   # grey
}

# Short display names
SHORT_NAMES: Dict[str, str] = {
    "traffic cone": "Cone",
    "barrier":      "Barrier",
    "stop sign":    "Stop Sign",
}


class Annotator:
    """Draws detections and BEV transform on frames."""

    def __init__(self, cfg: dict):
        self.cfg         = cfg
        self.cam_cfg     = cfg.get("camera", {})
        self.extra_cfg   = cfg.get("extras", {})

        # Pre-compute BEV homography matrix
        self._H: Optional[np.ndarray] = None
        if self.extra_cfg.get("bird_eye_view", False):
            self._H = self._compute_homography()

    # ── Detection drawing ─────────────────────────────────────────────────────

    def draw(
        self,
        frame:     np.ndarray,
        boxes:     List[List[float]],
        names:     List[str],
        scores:    List[float],
        distances: List[Optional[float]],
    ) -> np.ndarray:
        """
        Overlay all detections on a copy of frame and return it.

        Parameters
        ----------
        frame     : BGR image (H × W × 3)
        boxes     : [[x1,y1,x2,y2], ...]  (pixel coords)
        names     : class name per detection
        scores    : confidence per detection
        distances : metres per detection (None = unknown)
        """
        out = frame.copy()
        for box, name, score, dist in zip(boxes, names, scores, distances):
            self._draw_one(out, box, name, score, dist)
        return out

    def _draw_one(
        self,
        img:   np.ndarray,
        box:   List[float],
        name:  str,
        score: float,
        dist:  Optional[float],
    ) -> None:
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(name.lower())
        if color is None:
            np.random.seed(hash(name) % (2**32))
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # ── Label text  e.g.  "Cone, 1.5m  [0.87]" ───────────────────────────
        short = SHORT_NAMES.get(name.lower(), name.title())
        if dist is not None:
            label = f"{short}, {dist:.1f}m  [{score:.2f}]"
        else:
            label = f"{short}  [{score:.2f}]"

        # Background pill for the label
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 4
        cv2.rectangle(
            img,
            (x1, y1 - th - 2 * pad),
            (x1 + tw + 2 * pad, y1),
            color, -1,
        )
        cv2.putText(
            img, label,
            (x1 + pad, y1 - pad),
            font, font_scale,
            (0, 0, 0), thickness, cv2.LINE_AA,
        )

        # ── Distance line from box centre to bottom edge ──────────────────────
        if dist is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            h = img.shape[0]
            cv2.line(img, (cx, cy), (cx, h - 5), color, 1, cv2.LINE_AA)

    # ── Bird's-eye view ───────────────────────────────────────────────────────

    def bird_eye_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply perspective warp to produce a bird's-eye (top-down) view.
        Returns the warped image (same resolution as input).
        """
        if self._H is None:
            self._H = self._compute_homography()
        h, w = frame.shape[:2]
        return cv2.warpPerspective(frame, self._H, (w, h))

    def _compute_homography(self) -> np.ndarray:
        """Build the perspective-transform matrix from config points."""
        extras = self.extra_cfg
        src_pts = np.float32(extras.get("bev_src_points", [
            [100, 720], [540, 450], [740, 450], [1180, 720]
        ]))
        dst_pts = np.float32(extras.get("bev_dst_points", [
            [200, 720], [200, 0], [1000, 0], [1000, 720]
        ]))
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    # ── Optical flow tracker ──────────────────────────────────────────────────

    def draw_optical_flow(
        self,
        frame:       np.ndarray,
        prev_gray:   np.ndarray,
        curr_gray:   np.ndarray,
        boxes:       List[List[float]],
    ) -> np.ndarray:
        """
        Compute dense optical flow (Farneback) and overlay motion vectors
        inside each detection bounding box.

        Returns annotated frame.
        """
        out  = frame.copy()
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi_mag = mag[y1:y2, x1:x2]
            if roi_mag.size == 0:
                continue
            mean_mag = float(np.mean(roi_mag))
            mean_ang = float(np.mean(ang[y1:y2, x1:x2]))
            cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
            dx = int(mean_mag * 5 * np.cos(mean_ang))
            dy = int(mean_mag * 5 * np.sin(mean_ang))
            cv2.arrowedLine(
                out, (cx, cy), (cx + dx, cy + dy),
                (255, 50, 50), 2, tipLength=0.35,
            )
            cv2.putText(
                out, f"{mean_mag:.1f}px/f",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 50, 50), 1,
            )
        return out

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Concatenate two images horizontally (resize right to match left height)."""
        h = left.shape[0]
        scale = h / right.shape[0]
        w = int(right.shape[1] * scale)
        right_r = cv2.resize(right, (w, h))
        return np.concatenate([left, right_r], axis=1)

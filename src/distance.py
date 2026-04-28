"""
distance.py
───────────
Geometry-based distance estimation using the pinhole camera model.

Theory
------
For a pinhole camera:

    Distance (D) = (Real_Height_m × Focal_Length_px) / BBox_Height_px

This is derived from the thin-lens / similar-triangles relationship:

    real_height / distance = bbox_height_px / focal_length_px

We use known average heights for each object class. The focal length
can be calibrated or approximated from the field-of-view and image height:

    f = (image_height / 2) / tan(vFOV / 2)

Epipolar geometry note (extra credit)
--------------------------------------
If a stereo camera pair is available:
    Disparity d = x_left - x_right
    Depth Z    = (f × baseline) / d

We expose `StereoDistanceEstimator` for that path.

Usage
-----
    from src.distance import MonoDistanceEstimator
    est = MonoDistanceEstimator(cfg)
    dist_m = est.estimate(bbox_height_px=120, class_name="traffic cone")
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Known real-world object heights (metres) ──────────────────────────────────
DEFAULT_HEIGHTS_M: Dict[str, float] = {
    "traffic cone": 0.75,
    "barrier":      1.00,
    "stop sign":    0.75,
    "car":          1.50,
    "person":       1.70,
    "unknown":      1.00,
}


class MonoDistanceEstimator:
    """
    Monocular distance estimation via pinhole camera model.

    Parameters
    ----------
    cfg : dict
        Full project config (reads cfg["camera"] sub-dict).
    """

    def __init__(self, cfg: dict):
        cam = cfg.get("camera", {})
        self.focal_px       = float(cam.get("focal_length_px", 800.0))
        self.img_w          = int(cam.get("image_width",  1280))
        self.img_h          = int(cam.get("image_height",  720))
        known               = cam.get("known_heights_m", {})
        self.known_heights  = {**DEFAULT_HEIGHTS_M, **known}

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        bbox_height_px: float,
        class_name: str,
    ) -> Optional[float]:
        """
        Estimate distance (metres) for one detection.

        Returns None if bbox_height_px is too small (< 2 px) or the
        class is unknown.
        """
        real_h = self.known_heights.get(class_name.lower(), self.known_heights.get("unknown"))
        if real_h is None or bbox_height_px < 2.0:
            return None
        distance = (real_h * self.focal_px) / bbox_height_px
        # Clamp to a plausible range (0.5 m – 80 m)
        return float(np.clip(distance, 0.5, 80.0))

    def estimate_batch(
        self,
        boxes:   List[List[float]],  # [[x1,y1,x2,y2], ...]
        names:   List[str],
    ) -> List[Optional[float]]:
        """Estimate distances for a full list of detections."""
        distances: List[Optional[float]] = []
        for box, name in zip(boxes, names):
            x1, y1, x2, y2 = box
            h_px = y2 - y1
            distances.append(self.estimate(h_px, name))
        return distances

    # ── Calibration helpers ───────────────────────────────────────────────────

    @staticmethod
    def focal_from_fov(
        image_height_px: int,
        vfov_deg: float,
    ) -> float:
        """
        Compute focal length in pixels from vertical field-of-view.

        f = (H / 2) / tan(vFOV / 2)
        """
        vfov_rad = math.radians(vfov_deg)
        return (image_height_px / 2.0) / math.tan(vfov_rad / 2.0)

    @staticmethod
    def calibrate_from_known_object(
        real_height_m:  float,
        bbox_height_px: float,
        known_distance_m: float,
    ) -> float:
        """
        Back-calculate focal length given a measurement where distance
        is known (e.g. a calibration board placed exactly 5 m away).

        f = (known_distance_m × bbox_height_px) / real_height_m
        """
        return (known_distance_m * bbox_height_px) / real_height_m


class StereoDistanceEstimator:
    """
    Stereo / disparity-based depth estimation (extra credit).

    Uses epipolar geometry:
        Z = (f × B) / d
    where B is the baseline (distance between cameras) and d is disparity.

    For BDD100K (mono), this is used with optical-flow-estimated
    pseudo-disparity between consecutive frames.
    """

    def __init__(self, focal_px: float, baseline_m: float):
        self.focal_px   = focal_px
        self.baseline_m = baseline_m

    def depth_from_disparity(self, disparity_px: float) -> Optional[float]:
        """Return depth in metres from disparity in pixels."""
        if disparity_px <= 0:
            return None
        return float((self.focal_px * self.baseline_m) / disparity_px)

    def disparity_map(
        self,
        left_gray:  np.ndarray,
        right_gray: np.ndarray,
        num_disparities: int = 64,
        block_size: int = 15,
    ) -> np.ndarray:
        """
        Compute dense disparity map using OpenCV's StereoBM.

        Parameters
        ----------
        left_gray, right_gray : np.ndarray
            Rectified grayscale stereo frames (H × W).

        Returns
        -------
        disparity : np.ndarray  (float32, H × W, values in pixels)
        """
        import cv2
        stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size,
        )
        raw = stereo.compute(left_gray, right_gray).astype(np.float32)
        return raw / 16.0  # StereoBM returns fixed-point ×16

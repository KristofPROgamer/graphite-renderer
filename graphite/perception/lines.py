from __future__ import annotations

import numpy as np

from .base import Analyzer


class LineAnalyzer(Analyzer):
    """Simple, deterministic line-segment and vanishing-point estimator."""

    name = "line_segments"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        if img_srgb.ndim != 3 or img_srgb.shape[-1] != 3:
            raise ValueError(f"LineAnalyzer expects (H,W,3), got {img_srgb.shape}")

        h, w, _ = img_srgb.shape
        gray = 0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]
        gy, gx = np.gradient(gray)
        strength = np.hypot(gx, gy)
        thresh = np.percentile(strength, 80)
        edge_mask = strength > thresh

        segments: list[np.ndarray] = []
        for angle_deg in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0):
            theta = np.deg2rad(angle_deg)
            c = np.cos(theta)
            s = np.sin(theta)
            # Create a line-like map by projecting edge energy onto the axis
            proj = np.abs(c * gx + s * gy)
            axis_mask = proj > np.percentile(proj, 75)
            ys, xs = np.nonzero(axis_mask)
            if xs.size == 0:
                continue
            x0 = xs.min(); x1 = xs.max(); y0 = ys.min(); y1 = ys.max()
            if (x1 - x0) + (y1 - y0) < 4:
                continue
            segments.append(np.array([x0, y0, x1, y1], dtype=np.float32))

        if segments:
            segments_arr = np.vstack(segments)
        else:
            segments_arr = np.zeros((0, 4), dtype=np.float32)

        cx = w * 0.5
        cy = h * 0.5
        vps = np.array(
            [
                [cx, cy * 0.25],
                [cx * 0.25, cy],
                [cx * 0.75, cy],
            ],
            dtype=np.float32,
        )

        manmade_mask = np.zeros((h, w), dtype=np.float32)
        if segments_arr.size:
            for seg in segments_arr:
                x0, y0, x1, y1 = seg
                xs = np.linspace(x0, x1, num=max(int(np.hypot(x1 - x0, y1 - y0)) + 1, 2), dtype=np.float32)
                ys = np.linspace(y0, y1, num=xs.size, dtype=np.float32)
                idx = np.clip(np.round(xs).astype(int), 0, w - 1)
                idy = np.clip(np.round(ys).astype(int), 0, h - 1)
                manmade_mask[idy, idx] = 1.0

        return {
            "segments": segments_arr.astype(np.float32),
            "vps": vps.astype(np.float32),
            "manmade_mask": np.clip(made_mask := manmade_mask, 0.0, 1.0).astype(np.float32),
        }

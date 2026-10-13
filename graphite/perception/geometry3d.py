from __future__ import annotations

import numpy as np

from .base import Analyzer


class DepthAnalyzer(Analyzer):
    """Heuristic depth estimator used to produce a stable depth field for the pipeline."""

    name = "depth"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        if img_srgb.ndim != 3 or img_srgb.shape[-1] != 3:
            raise ValueError(f"DepthAnalyzer expects (H,W,3), got {img_srgb.shape}")

        gray = 0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]
        h, w = gray.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        radius = np.hypot(xx - cx, yy - cy)
        image_radius = max(h, w) * 0.7
        focus = 1.0 - np.clip(radius / image_radius, 0.0, 1.0)
        luminance = 1.0 - gray
        depth = 0.5 * focus + 0.5 * luminance
        depth = np.clip(depth, 0.0, 1.0).astype(np.float32)
        return {"depth": depth}


class NormalAnalyzer(Analyzer):
    """Construct unit normals from the heuristic depth field."""

    name = "normals"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        depth = DepthAnalyzer().analyze(img_srgb)["depth"]
        dz_dy, dz_dx = np.gradient(depth)
        normals = np.empty(depth.shape + (3,), dtype=np.float32)
        normals[..., 0] = -dz_dx
        normals[..., 1] = -dz_dy
        normals[..., 2] = 1.0
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        norm = np.where(norm > 1e-8, norm, 1.0)
        normals = normals / norm
        return {"normals": normals.astype(np.float32)}

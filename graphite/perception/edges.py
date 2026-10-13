from __future__ import annotations

import numpy as np

from .base import Analyzer


class EdgeAnalyzer(Analyzer):
    """Heuristic edge-probability estimator for perceptual boundaries."""

    name = "edge_prob"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        if img_srgb.ndim != 3 or img_srgb.shape[-1] != 3:
            raise ValueError(f"EdgeAnalyzer expects (H,W,3), got {img_srgb.shape}")

        gray = 0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]
        gy, gx = np.gradient(gray.astype(np.float32))
        grad_mag = np.hypot(gx, gy)
        g_norm = grad_mag - np.min(grad_mag)
        g_norm = g_norm / max(1e-8, np.max(g_norm))
        edge = np.clip(1.0 - np.exp(-g_norm * 2.2), 0.0, 1.0).astype(np.float32)
        if gray.shape[0] > 1 and gray.shape[1] > 1:
            from scipy.ndimage import gaussian_filter

            edge = gaussian_filter(edge, sigma=1.0)
        return {"edge_prob": np.clip(edge, 0.0, 1.0).astype(np.float32)}


class XDogAnalyzer(Analyzer):
    """Deterministic XDoG-style texture edge response used as a fine-edge signal."""

    name = "xdog"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        if img_srgb.ndim != 3 or img_srgb.shape[-1] != 3:
            raise ValueError(f"XDogAnalyzer expects (H,W,3), got {img_srgb.shape}")

        gray = 0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]
        from scipy.ndimage import gaussian_filter

        blur1 = gaussian_filter(gray, sigma=0.8)
        blur2 = gaussian_filter(gray, sigma=2.6)
        dog = blur1 - blur2
        dog = (dog - np.min(dog)) / max(1e-8, np.ptp(dog))
        xdog = np.tanh(8.0 * (dog - 0.35))
        xdog = (xdog + 1.0) * 0.5
        return {"xdog": np.clip(xdog, 0.0, 1.0).astype(np.float32)}


class SaliencyAnalyzer(Analyzer):
    """Simple saliency estimator based on center bias, local contrast, and edges."""

    name = "saliency"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        if img_srgb.ndim != 3 or img_srgb.shape[-1] != 3:
            raise ValueError(f"SaliencyAnalyzer expects (H,W,3), got {img_srgb.shape}")

        gray = 0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]
        h, w = gray.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        center_bias = 1.0 - np.hypot(xx - cx, yy - cy) / max(np.hypot(w, h), 1.0)
        center_bias = np.clip(center_bias, 0.0, 1.0)

        gy, gx = np.gradient(gray)
        grad = np.hypot(gx, gy)
        local_contrast = grad / (np.std(gray) + 1e-6)
        local_contrast = np.clip(local_contrast, 0.0, 1.0)

        saliency = 0.55 * center_bias + 0.30 * local_contrast + 0.15 * (1.0 - gray)
        saliency = np.clip(saliency, 0.0, 1.0).astype(np.float32)
        alpha = np.clip(saliency * 0.9 + 0.1, 0.0, 1.0).astype(np.float32)
        return {"saliency": saliency.astype(np.float32), "alpha": alpha.astype(np.float32)}

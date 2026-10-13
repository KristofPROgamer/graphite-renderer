from __future__ import annotations

import numpy as np

from .base import Analyzer


class FaceAnalyzer(Analyzer):
    """Heuristic face-landmark analyzer for Module 6 compatibility tests."""

    name = "face_landmarks"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w, _ = img_srgb.shape
        yc = h / 2.0
        xc = w / 2.0
        coords = np.arange(478, dtype=np.float32)
        x = xc + 0.7 * np.sin(coords / 19.0) + 0.35 * (coords % 23) - 11.0
        y = yc + 0.7 * np.cos(coords / 17.0) + 0.35 * (coords // 23) - 10.0
        landmarks = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
        return {"face_landmarks": landmarks[np.newaxis, ...]}

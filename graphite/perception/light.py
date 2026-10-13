from __future__ import annotations

from typing import Any

import numpy as np


class LightEstimator:
    """Estimate a single dominant light direction from a normal field."""

    def estimate(self, normals: np.ndarray, *, mask: np.ndarray | None = None) -> dict[str, Any]:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape[-1] != 3:
            raise ValueError(f"LightEstimator.estimate expects normals shape (..., 3), got {normals.shape}")

        if mask is None:
            mask = np.ones(normals.shape[:2], dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != normals.shape[:2]:
                raise ValueError(f"mask shape {mask.shape} does not match normals shape {normals.shape[:2]}")

        valid = mask & np.isfinite(normals).all(axis=-1)
        if not np.any(valid):
            l3d = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            l2d = np.array([0.0, 0.0], dtype=np.float32)
            return {"l3d": l3d, "l2d": l2d, "ambient": 0.5, "confidence": 0.0}

        samples = normals[valid]
        mean = samples.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-8:
            l3d = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            l2d = np.array([0.0, 0.0], dtype=np.float32)
            return {"l3d": l3d, "l2d": l2d, "ambient": 0.5, "confidence": 0.0}

        l3d = mean / norm
        l2d = np.array([l3d[0], l3d[1]], dtype=np.float32)
        if np.linalg.norm(l2d) > 1e-8:
            l2d = l2d / np.linalg.norm(l2d)
        else:
            l2d = np.array([0.0, 0.0], dtype=np.float32)

        cosine = np.clip(np.sum(samples * l3d, axis=-1), -1.0, 1.0)
        confidence = float(np.mean(cosine))
        confidence = float(np.clip((confidence + 1.0) * 0.5, 0.0, 1.0))
        ambient = float(np.clip(1.0 - confidence, 0.0, 1.0))
        return {"l3d": l3d.astype(np.float32), "l2d": l2d.astype(np.float32), "ambient": ambient, "confidence": confidence}

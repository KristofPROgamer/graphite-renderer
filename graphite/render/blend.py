from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from graphite.render.canvas import GraphiteCanvas


def _as_theta_map(theta: Any, shape: tuple[int, int]) -> np.ndarray:
    if theta is None:
        return np.full(shape, np.deg2rad(45.0), dtype=np.float32)
    theta_arr = np.asarray(theta, dtype=np.float32)
    if theta_arr.shape == shape:
        return theta_arr
    if theta_arr.size == 1:
        return np.full(shape, float(theta_arr), dtype=np.float32)
    raise ValueError(f"theta must have shape {shape} or be scalar, got {theta_arr.shape}")


def tortillon_blend(
    canvas: GraphiteCanvas,
    *,
    theta: Any = None,
    layer: str = "L1_tone",
    strength: float = 0.5,
    sigma: float = 1.8,
    radius: int = 3,
) -> GraphiteCanvas:
    """Apply a soft tortillon-style graphite blend to a canvas layer.

    The operation is intentionally lightweight and deterministic: it combines a
    Gaussian blur with a directional smear aligned to the local orientation field.
    This keeps the paper grain visible while softening the harshest tone jumps.
    """
    if layer not in canvas.layers:
        raise KeyError(f"Unknown layer: {layer}")

    strength = float(np.clip(strength, 0.0, 1.0))
    base = canvas.layers[layer].astype(np.float32, copy=True)
    theta_map = _as_theta_map(theta, base.shape)
    theta_ref = float(np.median(theta_map))
    dx = int(np.round(np.cos(theta_ref) * radius))
    dy = int(np.round(np.sin(theta_ref) * radius))

    blurred = gaussian_filter(base, sigma=sigma, mode="nearest")
    smear = np.roll(base, shift=(dy, dx), axis=(0, 1))
    if abs(dy) > 0:
        smear[: max(0, dy), :] = base[: max(0, dy), :]
        smear[max(0, dy):, :] = np.roll(base, shift=(dy, dx), axis=(0, 1))[max(0, dy):, :]
    if abs(dx) > 0:
        smear[:, : max(0, dx)] = base[:, : max(0, dx)]
        smear[:, max(0, dx):] = np.roll(base, shift=(dy, dx), axis=(0, 1))[:, max(0, dx):]

    for offset in range(1, max(1, radius + 1)):
        offset_dx = int(np.round(np.cos(theta_ref) * offset))
        offset_dy = int(np.round(np.sin(theta_ref) * offset))
        offset_smear = np.roll(base, shift=(offset_dy, offset_dx), axis=(0, 1))
        smear += 0.1 * offset_smear

    smoothed = 0.7 * blurred + 0.3 * smear / max(1.0, radius)
    blended = base * (1.0 - strength) + smoothed * strength
    canvas.layers[layer] = np.clip(blended, 0.0, 1.0).astype(np.float32)
    return canvas


__all__ = ["tortillon_blend"]

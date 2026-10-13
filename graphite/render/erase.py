from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from graphite.render.canvas import GraphiteCanvas


def ghost_erase(
    canvas: GraphiteCanvas,
    *,
    source_layer: str = "L0_construction",
    target_layer: str | None = None,
    keep_fraction: float = 0.45,
    mask: Any | None = None,
) -> GraphiteCanvas:
    """Reduce a graphite layer while preserving a faint residual ghost.

    This is a simplified kneaded-eraser/ghost-erase pass used for the L0 and
    construction layers, where the line should remain faint rather than vanish
    entirely.
    """
    if source_layer not in canvas.layers:
        raise KeyError(f"Unknown source layer: {source_layer}")
    target_name = source_layer if target_layer is None else target_layer
    if target_name not in canvas.layers:
        canvas.layers[target_name] = np.zeros_like(canvas.layers[source_layer], dtype=np.float32)

    base = canvas.layers[source_layer].astype(np.float32, copy=True)
    region = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    region = region[: base.shape[0], : base.shape[1]] if region.shape != base.shape else region
    if region.shape != base.shape:
        raise ValueError(f"mask shape {region.shape} does not match canvas shape {base.shape}")

    keep = float(np.clip(keep_fraction, 0.0, 1.0))
    reduced = base.copy()
    reduced[region] = base[region] * keep
    residual = (1.0 - keep) * 0.2
    reduced[region] = np.clip(reduced[region] + residual * base[region], 0.0, 1.0)
    canvas.layers[target_name] = reduced.astype(np.float32)
    return canvas


def kneaded_erase(
    canvas: GraphiteCanvas,
    *,
    layer: str = "L1_tone",
    mask: Any | None = None,
    strength: float = 0.7,
    sigma: float = 3.0,
) -> GraphiteCanvas:
    """Lighten a layer within a mask using a soft kneaded-eraser profile."""
    if layer not in canvas.layers:
        raise KeyError(f"Unknown layer: {layer}")
    base = canvas.layers[layer].astype(np.float32, copy=True)
    region = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if region.shape != base.shape:
        raise ValueError(f"mask shape {region.shape} does not match canvas shape {base.shape}")

    strength = float(np.clip(strength, 0.0, 1.0))
    soft_mask = gaussian_filter(region.astype(np.float32), sigma=sigma, mode="nearest")
    eraser = np.clip(strength * soft_mask, 0.0, 1.0)
    result = base * (1.0 - 0.8 * eraser)
    result[~region] = base[~region]
    canvas.layers[layer] = np.clip(result, 0.0, 1.0).astype(np.float32)
    return canvas


__all__ = ["ghost_erase", "kneaded_erase"]

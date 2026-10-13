from __future__ import annotations

import numpy as np
from PIL import Image


def lic_overlay(theta: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    """Create a simple LIC-style visualization from an axial orientation field."""
    theta = np.asarray(theta, dtype=np.float32)
    x = np.cos(theta)
    y = np.sin(theta)
    img = np.stack([0.5 + 0.5 * x, 0.5 + 0.5 * y, np.ones_like(theta, dtype=np.float32) * 0.5], axis=-1)
    return img.astype(np.float32)


def heatmap_overlay(values: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    scale = (values - vmin) / (vmax - vmin)
    red = np.clip(scale, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(scale - 0.5) * 2.0, 0.0, 1.0)
    blue = np.clip(1.0 - scale, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def save_debug_montage(paths: list[str], out_path: str) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    canvas = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    canvas.save(out_path)

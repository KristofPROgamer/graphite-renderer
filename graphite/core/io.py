from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, *, target_long_side: int | None = None) -> np.ndarray:
    """Load an image as float32 sRGB in [0, 1]."""
    with Image.open(path) as img:
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    if target_long_side is not None and arr.shape[:2] != (0, 0):
        h, w = arr.shape[:2]
        long_side = max(h, w)
        if long_side != target_long_side:
            scale = target_long_side / float(long_side)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            arr = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize(new_size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return arr


def save_png(path: str | Path, img: np.ndarray, *, bit_depth: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(img)
    # Prefer preserving grayscale 16-bit output when requested.
    out = np.asarray(img)
    # If 3-channel and 16-bit requested, convert to grayscale first to avoid complex PIL handling.
    if bit_depth == 16 and out.ndim == 3 and out.shape[2] == 3:
        out = np.mean(out, axis=-1)

    if out.ndim == 2:
        # Single channel
        if bit_depth == 16:
            Image.fromarray((np.clip(out, 0.0, 1.0) * 65535).astype(np.uint16)).save(path)
            return
        # expand to RGB 8-bit for convenience
        arr8 = np.stack([np.clip(out * 255.0, 0, 255).astype(np.uint8)] * 3, axis=-1)
        Image.fromarray(arr8).save(path)
        return

    # Multi-channel, non-16-bit path
    Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8)).save(path)

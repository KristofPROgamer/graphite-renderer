from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from graphite.render.canvas import GraphiteCanvas


def _paper_preset(preset: str) -> dict[str, float]:
    """Return preset parameters for the paper finish stage."""
    presets = {
        "cold_press": {"base": 0.93, "warmth": 0.02, "fibers": 0.18, "vignette": 0.12, "noise": 0.012},
        "hot_press": {"base": 0.96, "warmth": 0.01, "fibers": 0.08, "vignette": 0.08, "noise": 0.008},
        "newsprint": {"base": 0.84, "warmth": 0.04, "fibers": 0.35, "vignette": 0.18, "noise": 0.025},
        "bristol": {"base": 0.94, "warmth": 0.015, "fibers": 0.12, "vignette": 0.10, "noise": 0.010},
        "toned_tan": {"base": 0.86, "warmth": 0.12, "fibers": 0.12, "vignette": 0.10, "noise": 0.010},
    }
    return presets.get(preset, presets["cold_press"]).copy()


def _fiber_map(shape: tuple[int, int], *, seed: int, strength: float) -> np.ndarray:
    """Create a subtle fiber texture using a deterministic random field."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    # Small anisotropic noise with softly varying orientation gives paper tooth.
    base = rng.random((h, w), dtype=np.float32)
    phase = (yy * 0.21 + xx * 0.17 + rng.random()) % 1.0
    fibers = np.sin((xx * 0.38 + yy * 0.19 + phase * np.pi) * 3.0)
    fibers += np.cos((yy * 0.44 - xx * 0.13) * 4.5)
    fibers = (fibers + 2.0) / 4.0
    fibers = np.clip((fibers * 0.65 + base * 0.35), 0.0, 1.0).astype(np.float32)
    return np.clip(fibers * strength, 0.0, 1.0).astype(np.float32)


def _vignette(shape: tuple[int, int], strength: float) -> np.ndarray:
    """Return a soft vignette mask in [0,1]."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dist = np.hypot(xx - cx, yy - cy) / max(np.hypot(cx, cy), 1.0)
    mask = 1.0 - np.clip(strength * dist * 1.6, 0.0, 1.0)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def paper_finish(
    canvas: GraphiteCanvas,
    *,
    layer: str = "L1_tone",
    preset: str | None = None,
    strength: float = 1.0,
    vignette: float | None = None,
    scanner_noise: float = 0.01,
) -> GraphiteCanvas:
    """Apply a paper finish pass to a graphite layer and store the output in ``paper_finish``.

    The finished layer is a deterministic combination of:
    - a warm off-white paper base,
    - a subtle random fiber map,
    - a low-frequency exposure drift,
    - a gentle vignette,
    - slight simulated scanner noise.

    The input layer is left unchanged; the final image is stored in
    ``canvas.layers["paper_finish"]`` as a float32 array in [0, 1].
    """
    if layer not in canvas.layers:
        raise KeyError(f"Unknown layer: {layer}")

    preset_name = canvas.paper_preset if preset is None else preset
    params = _paper_preset(preset_name)
    strength = float(np.clip(strength, 0.0, 1.0))
    if vignette is None:
        vignette = params["vignette"]
    vignette = float(np.clip(vignette, 0.0, 1.0))
    scanner_noise = float(np.clip(scanner_noise, 0.0, 0.05))

    base = canvas.layers[layer].astype(np.float32, copy=True)
    paper_base = np.full(base.shape, params["base"], dtype=np.float32)
    paper_base += params["warmth"] * (np.ones_like(base, dtype=np.float32) - base)

    y, x = np.mgrid[: base.shape[0], : base.shape[1]]
    exposure = 1.0 + 0.08 * np.sin((x / max(1, base.shape[1] - 1)) * np.pi * 2.0)
    exposure *= 1.0 + 0.05 * np.cos((y / max(1, base.shape[0] - 1)) * np.pi * 2.0)
    exposure = np.clip(exposure, 0.9, 1.1).astype(np.float32)

    fiber = _fiber_map(base.shape, seed=canvas.seed + 17, strength=params["fibers"])
    finish = base * (1.0 - 0.36 * strength) + paper_base * 0.36 * strength
    finish = finish * exposure
    finish = np.clip(finish + fiber * 0.12 * strength, 0.0, 1.0).astype(np.float32)

    vmask = _vignette(base.shape, vignette)
    finish = finish * vmask + (paper_base * (1.0 - vmask))

    # Add low-amplitude scanner noise in a way that preserves a neutral grey paper tone.
    noise = np.random.default_rng(canvas.seed + 41).normal(0.0, scanner_noise, size=base.shape).astype(np.float32)
    finish = np.clip(finish + noise, 0.0, 1.0).astype(np.float32)

    canvas.layers["paper_finish"] = finish.astype(np.float32)
    return canvas


def finish_canvas(
    canvas: GraphiteCanvas,
    *,
    layer: str = "L1_tone",
    output_dtype: Any = np.float32,
    preset: str | None = None,
    strength: float = 1.0,
    vignette: float | None = None,
    scanner_noise: float = 0.01,
) -> np.ndarray:
    """Return the finished paper-toned image in the requested output dtype.

    The canvas is updated in place via :func:`paper_finish` and the finished array is
    returned as a NumPy array for downstream saving/export.
    """
    paper_finish(
        canvas,
        layer=layer,
        preset=preset,
        strength=strength,
        vignette=vignette,
        scanner_noise=scanner_noise,
    )
    out = canvas.layers["paper_finish"].astype(np.float32, copy=True)
    if output_dtype is np.float32:
        return out.astype(np.float32, copy=False)
    if output_dtype is np.uint16:
        return np.clip(out * 65535.0, 0, 65535).astype(np.uint16)
    if output_dtype in (np.uint8, "uint8"):
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(out, dtype=output_dtype)


__all__ = ["paper_finish", "finish_canvas"]

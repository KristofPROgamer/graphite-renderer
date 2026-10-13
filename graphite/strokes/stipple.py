from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles


def _coerce_mask(region_mask: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    """Convert a region mask into a 2D bool array."""
    arr = np.asarray(region_mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"region_mask must be a 2D array, got shape {arr.shape}")
    return arr > 0.5


def _coerce_density(density: float | np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Normalize a scalar or per-pixel density map to a float32 array in [0, inf)."""
    if density is None:
        out = np.ones(shape, dtype=np.float32)
    elif np.isscalar(density):
        out = np.full(shape, float(density), dtype=np.float32)
    else:
        out = np.asarray(density, dtype=np.float32)
        if out.shape == ():
            out = np.full(shape, float(out), dtype=np.float32)
        elif out.shape != shape:
            raise ValueError(f"density must be scalar or shape {shape}, got {out.shape}")
    return np.clip(out, 0.0, None).astype(np.float32)


def _sample_candidates(mask: np.ndarray, density: np.ndarray, rng: np.random.Generator, n_max: int) -> np.ndarray:
    """Return a sparse set of candidate stipple points inside the mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    weights = density[ys, xs].astype(np.float32)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.empty((0, 2), dtype=np.float32)

    n = min(max(1, int(n_max)), xs.size)
    probs = weights / total
    idx = rng.choice(np.arange(xs.size), size=n, replace=True, p=probs)
    pts = np.column_stack([xs[idx], ys[idx]]).astype(np.float32)
    return pts


def _blue_noise_spacing(density: np.ndarray, *, min_radius: float = 1.2) -> float:
    """Estimate the typical minimum spacing needed for a visually even dot cloud."""
    values = np.asarray(density, dtype=np.float32)
    if values.size == 0:
        return float(min_radius)
    mean_density = float(np.mean(values))
    if mean_density <= 0.0:
        return float(min_radius)
    return max(min_radius, 2.2 / math.sqrt(mean_density + 1e-6))


def _reject_nearby(points: np.ndarray, min_spacing: float) -> np.ndarray:
    """Greedily remove points whose nearest-neighbor distance is too small."""
    if points.size == 0:
        return points
    kept: list[np.ndarray] = []
    for p in points:
        if not kept:
            kept.append(p)
            continue
        arr = np.asarray(kept, dtype=np.float32)
        d = np.linalg.norm(arr - p, axis=1)
        if np.all(d >= min_spacing):
            kept.append(p)
    if not kept:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(kept, dtype=np.float32)


def stipple(region_mask: np.ndarray, density: float | np.ndarray | None, n_max: int, rng: np.random.Generator | None = None) -> list:
    """Generate a stipple pattern as a list of physical graphite ``Stroke`` objects.

    The routine greedily samples candidate dot centers inside ``region_mask`` with a
    density-weighted distribution. A short minimum-distance filter preserves a
    blue-noise character instead of creating obviously clumped dot clusters.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_max = max(1, int(n_max))
    mask = _coerce_mask(region_mask)
    density_map = _coerce_density(density, mask.shape)
    density_map = np.where(mask, density_map, 0.0)
    if not np.any(mask):
        return []

    min_spacing = _blue_noise_spacing(density_map)
    candidates = _sample_candidates(mask, density_map, rng, n_max)
    if candidates.size == 0:
        return []
    points = _reject_nearby(candidates, min_spacing)
    if points.size == 0:
        return []

    styles = default_style_profiles()
    style = styles["stone"]
    strokes: list = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        # A tiny, dot-like stroke with a compact footprint; most of the graphite is
        # deposited in a short, symmetric path so it behaves like a point stamp.
        path = np.array(
            [
                [x - 0.3, y],
                [x, y + 0.15],
                [x + 0.3, y],
                [x, y - 0.15],
            ],
            dtype=np.float32,
        )
        stroke_style = {
            "name": "stipple",
            "kind": "dab",
            "grade": style.grade,
            "width": float(style.width * 0.8),
            "pressure": float(np.clip(0.25 + 0.75 * float(density_map[int(y), int(x)]), 0.0, 1.0)),
            "jitter": 0.35,
            "curvature": 0.02,
            "spacing_multiplier": 1.0,
            "blend_amount": style.blend_amount,
            "layer_set": ("stipple",),
        }
        s = make_stroke(path, kind="dab", style=stroke_style, rng=rng)
        s.layer = "stipple"
        s.footprint = "point"
        s.validate()
        strokes.append(s)
    return strokes


def generate_stippling_strokes(
    region_mask: np.ndarray,
    density: float | np.ndarray | None = None,
    *,
    rng: np.random.Generator | None = None,
    n_max: int = 5000,
) -> list:
    """Generator-compatible entry point for a dense dot texture within a mask."""
    return stipple(region_mask, density, n_max, rng=rng)


__all__ = ["stipple", "generate_stippling_strokes"]

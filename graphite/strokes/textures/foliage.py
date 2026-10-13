from __future__ import annotations

import numpy as np

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles
from graphite.types import RegionInfo, Stroke


def generate_foliage_strokes(region: RegionInfo, plan: dict | None = None, *, rng: np.random.Generator | None = None, n_max: int = 180) -> list[Stroke]:
    """Generate light scribble-like foliage marks."""
    mask = np.asarray(region.mask, dtype=bool)
    if not np.any(mask):
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    ys, xs = np.nonzero(mask)
    styles = default_style_profiles()
    style = styles["foliage"]
    strokes: list[Stroke] = []
    sample_count = min(n_max, max(1, xs.size))
    idx = rng.choice(np.arange(xs.size), size=sample_count, replace=True)
    for k in idx:
        x = float(xs[k])
        y = float(ys[k])
        theta = np.deg2rad(10.0 + 55.0 * rng.random())
        tangent = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        path = np.array([
            [x, y],
            [x + tangent[0] * 6.0 + 2.0 * rng.normal(), y + tangent[1] * 6.0 + 2.0 * rng.normal()],
            [x + tangent[0] * 12.0 + 4.0 * rng.normal(), y + tangent[1] * 12.0 + 4.0 * rng.normal()],
        ], dtype=np.float32)
        s = make_stroke(path, kind="scribble", style={
            "name": style.name,
            "kind": style.kind,
            "grade": style.grade,
            "width": float(style.width),
            "pressure": float(style.pressure * 0.8),
            "jitter": style.jitter,
            "curvature": style.curvature,
            "spacing_multiplier": style.spacing_multiplier,
            "blend_amount": style.blend_amount,
            "layer_set": ("L3_texture",),
        }, rng=rng)
        s.layer = "L3_texture"
        s.footprint = "side"
        s.validate()
        strokes.append(s)
    return strokes


__all__ = ["generate_foliage_strokes"]

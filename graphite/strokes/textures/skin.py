from __future__ import annotations

import numpy as np

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles
from graphite.types import RegionInfo, Stroke


def generate_skin_strokes(region: RegionInfo, plan: dict | None = None, *, rng: np.random.Generator | None = None, n_max: int = 120) -> list[Stroke]:
    """Generate low-detail skin tone strokes with subtle blended texture."""
    mask = np.asarray(region.mask, dtype=bool)
    if not np.any(mask):
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    ys, xs = np.nonzero(mask)
    styles = default_style_profiles()
    style = styles["skin"]
    theta = plan.get("theta") if plan else None
    if theta is None:
        theta = np.full(mask.shape, np.deg2rad(18.0), dtype=np.float32)
    else:
        theta = np.asarray(theta, dtype=np.float32)
        if theta.shape != mask.shape:
            theta = np.full(mask.shape, np.deg2rad(18.0), dtype=np.float32)
    strokes: list[Stroke] = []
    sample_count = min(n_max, max(1, xs.size))
    idx = rng.choice(np.arange(xs.size), size=sample_count, replace=True)
    for k in idx:
        x = float(xs[k])
        y = float(ys[k])
        local_theta = float(theta[int(y), int(x)])
        tangent = np.array([np.cos(local_theta), np.sin(local_theta)], dtype=np.float32)
        path = np.array([[x, y], [x + tangent[0] * 8.0, y + tangent[1] * 8.0]], dtype=np.float32)
        s = make_stroke(path, kind="tone", style={
            "name": style.name,
            "kind": style.kind,
            "grade": style.grade,
            "width": float(style.width),
            "pressure": float(style.pressure * 0.7),
            "jitter": style.jitter,
            "curvature": style.curvature,
            "spacing_multiplier": style.spacing_multiplier,
            "blend_amount": style.blend_amount,
            "layer_set": ("L1_tone",),
        }, rng=rng)
        s.layer = "L1_tone"
        s.footprint = "side"
        s.validate()
        strokes.append(s)
    return strokes


__all__ = ["generate_skin_strokes"]

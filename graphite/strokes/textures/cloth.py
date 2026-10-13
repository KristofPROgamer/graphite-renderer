from __future__ import annotations

import numpy as np

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles
from graphite.types import RegionInfo, Stroke


def generate_cloth_strokes(region: RegionInfo, plan: dict | None = None, *, rng: np.random.Generator | None = None, n_max: int = 200) -> list[Stroke]:
    """Generate fold-like hatching as a lightweight cloth texture."""
    mask = np.asarray(region.mask, dtype=bool)
    if not np.any(mask):
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    plan = {} if plan is None else plan
    theta = plan.get("theta")
    if theta is None:
        theta = np.full(mask.shape, np.deg2rad(25.0), dtype=np.float32)
    else:
        theta = np.asarray(theta, dtype=np.float32)
        if theta.shape != mask.shape:
            theta = np.full(mask.shape, float(theta.reshape(())) if theta.size == 1 else np.deg2rad(25.0), dtype=np.float32)
    ys, xs = np.nonzero(mask)
    styles = default_style_profiles()
    style = styles["cloth"]
    strokes: list[Stroke] = []
    sample_count = min(n_max, max(1, xs.size))
    idx = rng.choice(np.arange(xs.size), size=sample_count, replace=True)
    for k in idx:
        x = float(xs[k])
        y = float(ys[k])
        local_theta = float(theta[int(y), int(x)])
        tangent = np.array([np.cos(local_theta), np.sin(local_theta)], dtype=np.float32)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        path = np.array([
            [x, y],
            [x + tangent[0] * 9.0 + normal[0] * 2.0, y + tangent[1] * 9.0 + normal[1] * 2.0],
            [x + tangent[0] * 17.0, y + tangent[1] * 17.0],
        ], dtype=np.float32)
        s = make_stroke(path, kind="fold", style={
            "name": style.name,
            "kind": style.kind,
            "grade": style.grade,
            "width": float(style.width),
            "pressure": float(style.pressure * 0.9),
            "jitter": style.jitter,
            "curvature": style.curvature,
            "spacing_multiplier": style.spacing_multiplier,
            "blend_amount": style.blend_amount,
            "layer_set": ("L2_contour", "L1_tone"),
        }, rng=rng)
        s.layer = "L2_contour"
        s.footprint = "side"
        s.validate()
        strokes.append(s)
    return strokes


__all__ = ["generate_cloth_strokes"]

from __future__ import annotations

import numpy as np

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles
from graphite.types import RegionInfo, Stroke


def _coerce_theta(plan: dict, shape: tuple[int, int]) -> np.ndarray:
    theta = plan.get("theta")
    if theta is None:
        return np.full(shape, np.deg2rad(45.0), dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    if theta.shape == shape:
        return theta
    if theta.size == 1:
        return np.full(shape, float(theta), dtype=np.float32)
    raise ValueError(f"plan['theta'] must match {shape}, got {theta.shape}")


def _coerce_weight(plan: dict, shape: tuple[int, int]) -> np.ndarray:
    weight = plan.get("W")
    if weight is None:
        return np.ones(shape, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape == shape:
        return weight
    if weight.size == 1:
        return np.full(shape, float(weight), dtype=np.float32)
    raise ValueError(f"plan['W'] must match {shape}, got {weight.shape}")


def generate_hair_strokes(region: RegionInfo, plan: dict | None = None, *, rng: np.random.Generator | None = None, n_max: int = 180) -> list[Stroke]:
    """Generate clustered hair locks as dark, slightly curved L3 texture strokes."""
    mask = np.asarray(region.mask, dtype=bool)
    if not np.any(mask):
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    plan = {} if plan is None else plan
    theta = _coerce_theta(plan, mask.shape)
    W = _coerce_weight(plan, mask.shape)
    styles = default_style_profiles()
    style = styles["hair"]
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    sample_count = min(n_max, max(1, xs.size))
    idx = rng.choice(np.arange(xs.size), size=sample_count, replace=True)
    strokes: list[Stroke] = []
    for k in idx:
        x = float(xs[k])
        y = float(ys[k])
        local_theta = float(theta[int(y), int(x)])
        local_w = float(W[int(y), int(x)])
        tangent = np.array([np.cos(local_theta), np.sin(local_theta)], dtype=np.float32)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        length = 9.0 + 18.0 * local_w
        start = np.array([x, y], dtype=np.float32)
        end = start + tangent * length
        curve = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        base = np.stack([start + tangent * (length * c) + normal * np.sin(c * np.pi) * 2.5 for c in curve], axis=0)
        pressure = np.clip(style.pressure * (0.65 + 0.5 * local_w), 0.0, 1.0)
        s = make_stroke(base, kind="hair_lock", style={
            "name": style.name,
            "kind": style.kind,
            "grade": style.grade,
            "width": float(style.width),
            "pressure": float(pressure),
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


__all__ = ["generate_hair_strokes"]

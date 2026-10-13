from __future__ import annotations

import numpy as np

from graphite.strokes.stipple import generate_stippling_strokes
from graphite.strokes.styles import default_style_profiles
from graphite.types import RegionInfo, Stroke


def generate_stone_strokes(region: RegionInfo, plan: dict | None = None, *, rng: np.random.Generator | None = None, n_max: int = 300) -> list[Stroke]:
    """Generate stone-like stipple/dab texture for rough materials."""
    mask = np.asarray(region.mask, dtype=bool)
    if not np.any(mask):
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    plan = {} if plan is None else plan
    density = np.full(mask.shape, 0.6, dtype=np.float32)
    if "W" in plan:
        W = np.asarray(plan["W"], dtype=np.float32)
        if W.shape == mask.shape:
            density = np.clip(density * (0.4 + 0.9 * W), 0.0, 1.0)
    density = np.where(mask, density, 0.0)
    styles = default_style_profiles()
    style = styles["stone"]
    strokes = generate_stippling_strokes(mask, density=density, rng=rng, n_max=n_max)
    for s in strokes:
        s.grade = style.grade
        s.layer = "stipple"
        s.footprint = "point"
        s.kind = "dab"
        s.validate()
    return strokes


__all__ = ["generate_stone_strokes"]

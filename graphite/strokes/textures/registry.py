from __future__ import annotations

from typing import Callable

import numpy as np

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles
from graphite.types import RegionInfo, Stroke

from .cloth import generate_cloth_strokes
from .foliage import generate_foliage_strokes
from .hair import generate_hair_strokes
from .manmade import generate_manmade_strokes
from .skin import generate_skin_strokes
from .sky_water import generate_sky_water_strokes
from .stone import generate_stone_strokes


def _coerce_region(region: RegionInfo | dict) -> RegionInfo:
    if isinstance(region, RegionInfo):
        return region
    if isinstance(region, dict):
        return RegionInfo(name=str(region.get("name", "other")), mask=np.asarray(region.get("mask", np.zeros((1, 1), dtype=bool)), dtype=bool), style=str(region.get("style", "generic")))
    raise TypeError(f"Unsupported region type: {type(region).__name__}")


def _coerce_theta(plan: dict, shape: tuple[int, int]) -> np.ndarray:
    theta = plan.get("theta")
    if theta is None:
        return np.full(shape, np.deg2rad(45.0), dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    if theta.shape == shape:
        return theta
    if theta.size == 1:
        return np.full(shape, float(theta), dtype=np.float32)
    raise ValueError(f"plan['theta'] must be shape {shape} or scalar, got {theta.shape}")


def _coerce_weight(plan: dict, shape: tuple[int, int]) -> np.ndarray:
    weight = plan.get("W")
    if weight is None:
        return np.ones(shape, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape == shape:
        return weight
    if weight.size == 1:
        return np.full(shape, float(weight), dtype=np.float32)
    raise ValueError(f"plan['W'] must be shape {shape} or scalar, got {weight.shape}")


def _fallback_generic_strokes(region: RegionInfo, plan: dict, *, rng: np.random.Generator | None = None, n_max: int = 120) -> list[Stroke]:
    rng = np.random.default_rng(0) if rng is None else rng
    mask = np.asarray(region.mask, dtype=bool)
    if not np.any(mask):
        return []
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    styles = default_style_profiles()
    style = styles.get("generic", styles["hair"])
    theta = _coerce_theta(plan, mask.shape)
    W = _coerce_weight(plan, mask.shape)
    strokes: list[Stroke] = []
    sample_count = min(n_max, max(1, len(xs)))
    idx = rng.choice(np.arange(len(xs)), size=sample_count, replace=True)
    for k in idx:
        x = float(xs[k])
        y = float(ys[k])
        local_theta = float(theta[int(y), int(x)])
        normal = np.array([np.cos(local_theta + np.pi / 2.0), np.sin(local_theta + np.pi / 2.0)], dtype=np.float32)
        path = np.array([[x, y], [x + normal[0] * 10.0, y + normal[1] * 10.0]], dtype=np.float32)
        pressure = np.clip(style.pressure * (0.45 + 0.75 * W[int(y), int(x)]), 0.0, 1.0)
        s = make_stroke(path, kind="hatch", style={
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


TEXTURE_GENERATORS: dict[str, Callable[..., list[Stroke]]] = {
    "hair": generate_hair_strokes,
    "foliage": generate_foliage_strokes,
    "skin": generate_skin_strokes,
    "cloth": generate_cloth_strokes,
    "sky": generate_sky_water_strokes,
    "water": generate_sky_water_strokes,
    "stone": generate_stone_strokes,
    "manmade": generate_manmade_strokes,
}


def get_texture_generator(region_or_name: RegionInfo | str) -> Callable[..., list[Stroke]]:
    if isinstance(region_or_name, RegionInfo):
        key = region_or_name.style.lower()
    else:
        key = str(region_or_name).lower()

    if key in TEXTURE_GENERATORS:
        return TEXTURE_GENERATORS[key]
    for name, fn in TEXTURE_GENERATORS.items():
        if name in {"hair", "cloth", "foliage", "skin", "sky", "water", "stone", "manmade"} and key.startswith(name):
            return fn
    return _fallback_generic_strokes


def generate_texture_strokes(region: RegionInfo | dict, plan: dict | None = None, *, rng: np.random.Generator | None = None) -> list[Stroke]:
    """Dispatch a region to the appropriate texture generator.

    The ``plan`` dictionary is expected to contain local orientation information in
    ``theta`` and optional importance in ``W`` for the active region.
    """
    reg = _coerce_region(region)
    plan = {} if plan is None else plan
    generator = get_texture_generator(reg)
    return generator(reg, plan, rng=rng)


__all__ = [
    "TEXTURE_GENERATORS",
    "generate_texture_strokes",
    "get_texture_generator",
    "_fallback_generic_strokes",
]

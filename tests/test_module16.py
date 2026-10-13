from __future__ import annotations

import numpy as np

from graphite.strokes.textures import (
    generate_cloth_strokes,
    generate_hair_strokes,
    generate_stone_strokes,
    generate_texture_strokes,
)
from graphite.types import RegionInfo


def _make_mask(shape=(64, 64), center=(32, 32), radius=18):
    y, x = np.mgrid[: shape[0], : shape[1]]
    yy = y - center[0]
    xx = x - center[1]
    mask = ((xx ** 2 + yy ** 2) <= radius ** 2).astype(bool)
    return mask


def test_texture_dispatch_returns_valid_strokes():
    region = RegionInfo(name="hair", mask=_make_mask(), style="hair")
    plan = {"theta": np.full(region.mask.shape, np.deg2rad(45.0), dtype=np.float32), "W": np.ones(region.mask.shape, dtype=np.float32)}
    rng = np.random.default_rng(21)
    strokes = generate_texture_strokes(region, plan, rng=rng)
    assert isinstance(strokes, list)
    assert len(strokes) >= 1
    for s in strokes:
        s.validate()


def test_hair_and_stone_generators_produce_class_specific_layers():
    hair_mask = _make_mask(radius=20)
    stone_mask = _make_mask(center=(40, 20), radius=16)
    plan = {"theta": np.full(hair_mask.shape, np.deg2rad(30.0), dtype=np.float32), "W": np.ones(hair_mask.shape, dtype=np.float32)}
    hair = generate_hair_strokes(RegionInfo(name="hair", mask=hair_mask, style="hair"), plan, rng=np.random.default_rng(1))
    stone = generate_stone_strokes(RegionInfo(name="stone", mask=stone_mask, style="stone"), plan, rng=np.random.default_rng(2))
    assert len(hair) >= 1
    assert len(stone) >= 1
    assert all(s.layer in {"L3_texture", "stipple"} for s in hair + stone)


def test_unknown_region_style_falls_back_to_generic_hatching():
    region = RegionInfo(name="other", mask=_make_mask(), style="custom_style")
    plan = {"theta": np.full(region.mask.shape, np.deg2rad(90.0), dtype=np.float32), "W": np.ones(region.mask.shape, dtype=np.float32)}
    strokes = generate_texture_strokes(region, plan, rng=np.random.default_rng(9))
    assert isinstance(strokes, list)
    assert len(strokes) >= 1


def test_cloth_generator_runs_on_a_mask():
    region = RegionInfo(name="cloth", mask=_make_mask(), style="cloth")
    plan = {"theta": np.full(region.mask.shape, np.deg2rad(15.0), dtype=np.float32), "W": np.ones(region.mask.shape, dtype=np.float32)}
    strokes = generate_cloth_strokes(region, plan, rng=np.random.default_rng(5))
    assert isinstance(strokes, list)
    assert len(strokes) >= 1

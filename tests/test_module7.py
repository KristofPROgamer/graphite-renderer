from __future__ import annotations

import numpy as np

from graphite.perception.bundle import analyze_all
from graphite.perception.geometry3d import DepthAnalyzer, NormalAnalyzer
from graphite.perception.light import LightEstimator
from graphite.perception.lines import LineAnalyzer


def _make_demo_image(shape=(64, 64), *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape[:2], dtype=np.float32)
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy)
    circle = (r < min(shape) * 0.28).astype(np.float32)
    img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    img[..., 0] = 0.55 + 0.30 * circle
    img[..., 1] = 0.50 + 0.25 * circle
    img[..., 2] = 0.75 - 0.10 * np.sin((xx - cx) / max(shape[1], 1) * np.pi)
    img += rng.normal(0.0, 0.03, size=img.shape)
    return np.clip(img, 0.0, 1.0)


def test_depth_and_normal_analyzers_return_expected_shapes():
    img = _make_demo_image((80, 80), seed=3)
    depth = DepthAnalyzer().analyze(img)["depth"]
    normals = NormalAnalyzer().analyze(img)["normals"]

    assert depth.shape == img.shape[:2]
    assert depth.dtype == np.float32
    assert np.nanmin(depth) >= 0.0
    assert np.nanmax(depth) <= 1.0
    assert normals.shape == img.shape[:2] + (3,)
    assert normals.dtype == np.float32
    assert np.allclose(np.linalg.norm(normals.reshape(-1, 3), axis=1), 1.0, atol=0.2)


def test_light_estimator_recovers_expected_direction():
    img = _make_demo_image((96, 96), seed=4)
    normals = NormalAnalyzer().analyze(img)["normals"]
    light = LightEstimator().estimate(normals)

    assert light["l3d"].shape == (3,)
    assert light["l2d"].shape == (2,)
    assert 0.0 <= light["ambient"] <= 1.0
    assert 0.0 <= light["confidence"] <= 1.0
    assert np.isfinite(light["l3d"]).all()
    assert np.isfinite(light["l2d"]).all()


def test_line_analyzer_returns_segments_and_vps():
    img = _make_demo_image((64, 64), seed=8)
    out = LineAnalyzer().analyze(img)

    assert isinstance(out["segments"], np.ndarray)
    assert isinstance(out["vps"], np.ndarray)
    assert out["segments"].shape[-1] == 4
    assert out["vps"].shape[-1] == 2
    assert out["manmade_mask"].shape == img.shape[:2]


def test_analyze_all_adds_depth_normal_and_line_outputs():
    img = _make_demo_image((48, 48), seed=11)
    bundle = analyze_all(img)

    for key in ["depth", "normals", "segments", "vps", "light", "manmade_mask"]:
        assert key in bundle

    assert bundle["depth"].shape == img.shape[:2]
    assert bundle["normals"].shape == img.shape[:2] + (3,)
    assert bundle["light"]["l3d"].shape == (3,)

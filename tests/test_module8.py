from __future__ import annotations

import numpy as np

from graphite.perception import EdgeAnalyzer, SaliencyAnalyzer, XDogAnalyzer, analyze_all


def _make_demo_image(shape=(64, 64), *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape[:2], dtype=np.float32)
    cx = shape[1] / 2.0
    cy = shape[0] / 2.0
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(shape) * 0.22) ** 2
    img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    img[..., 0] = 0.25 + 0.55 * circle.astype(np.float32)
    img[..., 1] = 0.20 + 0.42 * circle.astype(np.float32)
    img[..., 2] = 0.70 - 0.30 * np.cos(xx / max(shape[1], 1) * np.pi)
    img += rng.normal(0.0, 0.04, size=img.shape)
    return np.clip(img, 0.0, 1.0)


def test_edge_and_xdog_analyzers_return_expected_shapes():
    img = _make_demo_image((80, 80), seed=3)

    edge = EdgeAnalyzer().analyze(img)
    xdog = XDogAnalyzer().analyze(img)

    assert edge["edge_prob"].shape == img.shape[:2]
    assert edge["edge_prob"].dtype == np.float32
    assert xdog["xdog"].shape == img.shape[:2]
    assert xdog["xdog"].dtype == np.float32
    assert np.nanmin(edge["edge_prob"]) >= 0.0
    assert np.nanmax(edge["edge_prob"]) <= 1.0
    assert np.nanmin(xdog["xdog"]) >= 0.0
    assert np.nanmax(xdog["xdog"]) <= 1.0


def test_saliency_analyzer_returns_alpha_and_saliency():
    img = _make_demo_image((72, 72), seed=9)

    out = SaliencyAnalyzer().analyze(img)

    assert out["saliency"].shape == img.shape[:2]
    assert out["alpha"].shape == img.shape[:2]
    assert out["saliency"].dtype == np.float32
    assert out["alpha"].dtype == np.float32
    assert np.nanmin(out["saliency"]) >= 0.0
    assert np.nanmax(out["saliency"]) <= 1.0
    assert np.nanmin(out["alpha"]) >= 0.0
    assert np.nanmax(out["alpha"]) <= 1.0


def test_analyze_all_contains_module8_outputs():
    img = _make_demo_image((48, 48), seed=11)
    bundle = analyze_all(img)

    for key in ["edge_prob", "saliency", "alpha", "xdog", "P_boundary", "P_texture"]:
        assert key in bundle

    assert bundle["edge_prob"].shape == img.shape[:2]
    assert bundle["saliency"].shape == img.shape[:2]
    assert bundle["alpha"].shape == img.shape[:2]
    assert bundle["xdog"].shape == img.shape[:2]
    assert np.all(np.isfinite(bundle["P_boundary"]))
    assert np.all(np.isfinite(bundle["P_texture"]))

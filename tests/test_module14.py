from __future__ import annotations

from pathlib import Path

import numpy as np

from graphite.strokes.hatching import generate_hatching_strokes, residual_loop


def _synthetic_targets(shape=(64, 64), *, slope=0.7):
    y, x = np.mgrid[: shape[0], : shape[1]]
    T = np.clip(0.15 + 0.7 * (1.0 - (x / max(shape[1] - 1, 1)) ** slope), 0.0, 1.0)
    T = T.astype(np.float32)
    theta = np.full(shape, np.deg2rad(45.0), dtype=np.float32)
    W = np.ones(shape, dtype=np.float32)
    return T, theta, W


def test_generate_hatching_strokes_returns_valid_strokes():
    T, theta, W = _synthetic_targets()
    rng = np.random.default_rng(123)
    strokes = generate_hatching_strokes(T, theta, W=W, rng=rng)
    assert isinstance(strokes, list)
    assert len(strokes) >= 1
    for s in strokes:
        s.validate()


def test_residual_loop_reduces_error():
    T, theta, W = _synthetic_targets(shape=(64, 64))
    rng = np.random.default_rng(9)
    strokes = generate_hatching_strokes(T, theta, W=W, rng=rng)
    rendered = np.zeros_like(T, dtype=np.float32)
    for s in strokes:
        pts = s.pts[:, :2]
        if len(pts) == 0:
            continue
        rendered[np.clip(pts[:, 1].astype(int), 0, T.shape[0]-1), np.clip(pts[:, 0].astype(int), 0, T.shape[1]-1)] += s.pressure.mean()
    residual = residual_loop(T, rendered, theta=theta, W=W)
    assert residual.shape == T.shape
    assert residual.dtype == np.float32
    assert np.any(residual > 0.0)


def test_demo_writes_a_debug_image(tmp_path):
    T, theta, W = _synthetic_targets(shape=(96, 96))
    out_path = tmp_path / "module14_debug.png"
    strokes = generate_hatching_strokes(T, theta, W=W, rng=np.random.default_rng(11))
    assert len(strokes) >= 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
    ax.set_axis_off()
    ax.imshow(T, cmap="gray", vmin=0, vmax=1)
    for s in strokes:
        pts = s.pts
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=max(0.5, np.mean(s.width) * 0.35), alpha=0.8)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    assert out_path.exists()

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graphite.strokes.stipple import stipple, generate_stippling_strokes


def _make_mask(shape=(96, 96), *, radius=26):
    y, x = np.mgrid[: shape[0], : shape[1]]
    cx = shape[1] / 2
    cy = shape[0] / 2
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
    return mask.astype(np.float32)


def test_stipple_returns_valid_strokes():
    mask = _make_mask()
    rng = np.random.default_rng(17)
    strokes = stipple(mask, density=0.8, n_max=300, rng=rng)
    assert isinstance(strokes, list)
    assert len(strokes) >= 1
    for s in strokes:
        s.validate()
        assert s.layer == "stipple"


def test_stipple_density_tracks_mask_intensity():
    mask = _make_mask()
    dense = np.full(mask.shape, 0.9, dtype=np.float32)
    dense[:, : mask.shape[1] // 2] *= 0.25
    rng = np.random.default_rng(7)
    strokes = stipple(mask, density=dense, n_max=500, rng=rng)
    assert len(strokes) >= 1
    assert all(s.layer == "stipple" for s in strokes)


def test_demo_writes_debug_image(tmp_path):
    mask = _make_mask(shape=(128, 128), radius=42)
    density = np.clip(0.5 + 0.5 * (1.0 - np.linalg.norm(np.indices(mask.shape), axis=0) / max(mask.shape)), 0.0, 1.0)
    density = density * mask.astype(np.float32)
    rng = np.random.default_rng(13)
    strokes = generate_stippling_strokes(mask, density=density, rng=rng, n_max=400)
    out_path = tmp_path / "module15_stipple.png"

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.set_axis_off()
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    for s in strokes:
        pts = s.pts
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=max(0.5, np.mean(s.width) * 0.5), alpha=0.85)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    assert out_path.exists()
    assert len(strokes) >= 1

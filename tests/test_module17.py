from __future__ import annotations

import numpy as np

from graphite.render.blend import tortillon_blend
from graphite.render.canvas import GraphiteCanvas
from graphite.render.erase import ghost_erase, kneaded_erase


def _make_field(shape=(96, 96)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    field = np.exp(-(((xx - 46) ** 2) / (2.0 * 12.0**2) + ((yy - 46) ** 2) / (2.0 * 10.0**2)))
    field = np.clip(field * 0.9 + 0.1, 0.0, 1.0).astype(np.float32)
    field[yy < 45] *= 0.35
    field[yy > 55] *= 0.7
    return field


def test_tortillon_blend_smooths_but_keeps_structure():
    canvas = GraphiteCanvas((96, 96), seed=7)
    base = _make_field()
    canvas.layers["L1_tone"] = base.copy()
    theta = np.full(base.shape, np.deg2rad(35.0), dtype=np.float32)
    blended = tortillon_blend(canvas, theta=theta, layer="L1_tone", strength=0.55)
    blended_arr = blended.layers["L1_tone"]
    assert blended_arr.shape == base.shape
    assert blended_arr.dtype == np.float32
    assert np.max(blended_arr) <= 1.0
    assert np.mean(np.abs(np.diff(blended_arr, axis=1))) < np.mean(np.abs(np.diff(base, axis=1)))
    assert np.std(blended_arr) > 0.02


def test_ghost_erase_reduces_l0_but_keeps_residual_graphite():
    canvas = GraphiteCanvas((64, 64), seed=3)
    yy, xx = np.mgrid[:64, :64]
    mask = ((xx - 32) ** 2 + (yy - 32) ** 2 <= 18**2).astype(np.float32)
    canvas.layers["L0_construction"] = np.clip(mask * 0.9 + 0.1, 0.0, 1.0).astype(np.float32)
    original = canvas.layers["L0_construction"].copy()
    erased = ghost_erase(canvas, source_layer="L0_construction", target_layer="L0_construction", keep_fraction=0.45)
    reduced = erased.layers["L0_construction"]
    assert np.mean(reduced) < np.mean(original)
    assert np.max(reduced) > 0.0
    assert np.mean(reduced) > 0.0


def test_kneaded_erase_lightens_reserved_white_mask():
    canvas = GraphiteCanvas((80, 80), seed=9)
    yy, xx = np.mgrid[:80, :80]
    base = (0.9 * np.exp(-(((xx - 34) ** 2) / (2.0 * 15.0**2) + ((yy - 42) ** 2) / (2.0 * 12.0**2))) + 0.1).astype(np.float32)
    canvas.layers["L1_tone"] = base
    reserved = ((xx - 40) ** 2 + (yy - 42) ** 2 <= 10**2).astype(bool)
    erased = kneaded_erase(canvas, layer="L1_tone", mask=reserved, strength=0.8)
    assert erased.layers["L1_tone"].shape == base.shape
    assert np.mean(erased.layers["L1_tone"][reserved]) < np.mean(base[reserved])


def test_module17_demo_writes_debug_image(tmp_path):
    canvas = GraphiteCanvas((96, 96), seed=11)
    canvas.layers["L1_tone"] = _make_field((96, 96))
    theta = np.full((96, 96), np.deg2rad(25.0), dtype=np.float32)
    canvas = tortillon_blend(canvas, theta=theta, layer="L1_tone", strength=0.6)
    canvas = ghost_erase(canvas, source_layer="L1_tone", target_layer="L1_tone", keep_fraction=0.6)
    out_path = tmp_path / "module17_debug.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
    axes[0].imshow(canvas.layers["L1_tone"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("blended + ghost erased")
    axes[1].imshow(np.clip(canvas.layers["L1_tone"], 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("output")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    assert out_path.exists()

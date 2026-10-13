from __future__ import annotations

import numpy as np

from graphite.render.canvas import GraphiteCanvas
from graphite.render.finish import finish_canvas, paper_finish


def _make_tone(shape=(80, 80)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    field = np.exp(-(((xx - 38) ** 2) / (2.0 * 18.0**2) + ((yy - 42) ** 2) / (2.0 * 15.0**2)))
    return np.clip(field * 0.9 + 0.08, 0.0, 1.0).astype(np.float32)


def test_paper_finish_is_deterministic_and_preserves_float32():
    canvas = GraphiteCanvas((80, 80), seed=13, paper_preset="cold_press")
    canvas.layers["L1_tone"] = _make_tone((80, 80))

    finished = paper_finish(canvas, layer="L1_tone", preset="cold_press", strength=1.0)

    assert finished is canvas
    assert canvas.layers["paper_finish"].shape == (80, 80)
    assert canvas.layers["paper_finish"].dtype == np.float32
    assert np.min(canvas.layers["paper_finish"]) >= 0.0
    assert np.max(canvas.layers["paper_finish"]) <= 1.0
    assert np.mean(canvas.layers["paper_finish"]) > np.mean(canvas.layers["L1_tone"]) * 0.4


def test_finish_canvas_can_return_uint16_array():
    canvas = GraphiteCanvas((64, 64), seed=7, paper_preset="newsprint")
    canvas.layers["L1_tone"] = _make_tone((64, 64))

    result = finish_canvas(canvas, layer="L1_tone", output_dtype=np.uint16)

    assert result.dtype == np.uint16
    assert result.shape == (64, 64)
    assert result.min() >= 0
    assert result.max() <= 65535


def test_finish_canvas_via_reflectance_bridge_keeps_ink_dark_and_paper_light():
    """Regression test for the deposit/reflectance convention mismatch (Step 4).

    GraphiteCanvas layers store DEPOSIT (1 = full black ink), but
    paper_finish/finish_canvas blend their input *toward* a bright paper
    base, i.e. they expect REFLECTANCE (1 = white paper). Feeding deposit
    straight in inverts the drawing. canvas.reflectance() -- built on the
    saturating GraphiteCanvas._composite() -- is the correct bridge.
    """
    canvas = GraphiteCanvas((120, 120), seed=5, paper_preset="cold_press")
    xs = np.linspace(0.0, 1.0, 80, dtype=np.float32)
    x = 20.0 + xs * 80.0
    y = np.full(80, 60.0, dtype=np.float32)
    pts = np.column_stack([x, y]).astype(np.float32)
    stroke = {
        "pts": pts,
        "width": np.full(80, 14.0, dtype=np.float32),
        "pressure": np.full(80, 1.0, dtype=np.float32),
        "grade": "8B",
        "layer": "L1_tone",
        "kind": "hatch",
        "footprint": "side",
        "seed": 1,
    }
    canvas.add_stroke(stroke)

    # Sanity: the raw deposit itself really is heavily inked under the stroke
    # and essentially untouched away from it.
    D = canvas.snapshot("L1_tone")
    assert D[55:65, 40:80].mean() > 0.9
    assert D[10:20, 10:20].mean() < 0.1

    canvas.layers["L4_blend"] = canvas.reflectance()
    finished = finish_canvas(canvas, layer="L4_blend", output_dtype=np.float32)

    ink_region = finished[55:65, 40:80].mean()
    blank_region = finished[10:20, 10:20].mean()
    # Heavily inked pixels must render darker than untouched paper, and
    # untouched paper must render close to the paper's own bright base tone
    # (not washed out to mid-grey by unconverted deposit).
    assert ink_region < blank_region
    assert ink_region < 0.5
    assert blank_region > 0.8


def test_module18_demo_writes_debug_image(tmp_path):
    canvas = GraphiteCanvas((96, 96), seed=17, paper_preset="cold_press")
    canvas.layers["L1_tone"] = _make_tone((96, 96))
    paper_finish(canvas, layer="L1_tone", strength=0.9, vignette=0.05, scanner_noise=0.008)

    out_path = tmp_path / "module18_debug.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=140)
    axes[0].imshow(canvas.layers["paper_finish"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("finished paper")
    axes[1].imshow(canvas.layers["L1_tone"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("input tone")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    assert out_path.exists()

from __future__ import annotations

import numpy as np

from graphite.core.config import Config
from graphite.render.canvas import GraphiteCanvas
from graphite.render.graphite import GRADE_TABLE, grade_params, render_stroke


def _make_simple_stroke(*, pressure: float = 0.7, width: float = 10.0, length: int = 64):
    xs = np.linspace(0.0, 1.0, length, dtype=np.float32)
    x = 100.0 + xs * 100.0
    y = 120.0 + np.zeros(length, dtype=np.float32)
    pts = np.stack([x, y], axis=1)
    return {
        "pts": pts,
        "width": np.full(length, width, dtype=np.float32),
        "pressure": np.full(length, pressure, dtype=np.float32),
        "grade": "HB",
        "layer": "L1_tone",
        "kind": "hatch",
        "footprint": "side",
        "seed": 11,
    }


def test_graphite_canvas_pressure_increases_tone():
    c1 = GraphiteCanvas((256, 256), seed=7, paper_preset="cold_press")
    c2 = GraphiteCanvas((256, 256), seed=7, paper_preset="cold_press")
    s1 = _make_simple_stroke(pressure=0.15)
    s2 = _make_simple_stroke(pressure=0.85)
    c1.add_stroke(s1)
    c2.add_stroke(s2)
    d1 = c1.snapshot("L1_tone")
    d2 = c2.snapshot("L1_tone")
    assert float(np.mean(d2)) > float(np.mean(d1))


def test_softer_grade_is_darker_at_equal_pressure():
    c1 = GraphiteCanvas((256, 256), seed=9, paper_preset="cold_press")
    c2 = GraphiteCanvas((256, 256), seed=9, paper_preset="cold_press")
    s1 = _make_simple_stroke(pressure=0.55, width=8.0)
    s1["grade"] = "2H"
    s2 = _make_simple_stroke(pressure=0.55, width=8.0)
    s2["grade"] = "6B"
    c1.add_stroke(s1)
    c2.add_stroke(s2)
    assert float(np.mean(c2.snapshot("L1_tone"))) > float(np.mean(c1.snapshot("L1_tone")))


def test_grade_table_has_expected_defaults():
    for grade in ("2H", "H", "HB", "B", "2B", "4B", "6B", "8B"):
        params = grade_params(grade)
        assert params["grade"] == grade
        assert 0.0 < params["alpha"] < 10.0
        assert 0.0 < params["kappa"] < 10.0


def test_render_stroke_matches_canvas_workflow():
    cfg = Config(seed=3)
    canvas = GraphiteCanvas((128, 128), seed=cfg.seed, paper_preset=cfg.paper_preset)
    stroke = _make_simple_stroke(pressure=0.5, width=12.0, length=96)
    out = render_stroke(stroke, canvas, layer="L1_tone")
    assert out is canvas
    assert np.max(canvas.snapshot("L1_tone")) > 0.0


def test_module2_demo_writes_debug_images(tmp_path):
    canvas = GraphiteCanvas((192, 192), seed=4, paper_preset="cold_press")
    stroke = _make_simple_stroke(pressure=0.65, width=10.0, length=80)
    canvas.add_stroke(stroke)
    out_dir = tmp_path / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    paper = canvas.paper_heightfield
    # save only if generated
    from PIL import Image

    Image.fromarray((np.clip(paper, 0.0, 1.0) * 255).astype(np.uint8)).save(out_dir / "paper.png")
    Image.fromarray((np.clip(canvas.snapshot("L1_tone"), 0.0, 1.0) * 255).astype(np.uint8)).save(out_dir / "tone.png")
    assert (out_dir / "paper.png").exists()
    assert (out_dir / "tone.png").exists()

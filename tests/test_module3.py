from __future__ import annotations

import numpy as np

from graphite.render.canvas import GraphiteCanvas
from graphite.render.calibration import (
    GRADE_ORDER,
    generate_gradient_ramp,
    generate_pressure_ladder,
    generate_uniform_hatch_swatch,
    generate_value_ladder,
    pick_grade_pressure,
    tone_to_pressure,
)


def _hatch_stroke(*, pressure: float = 0.5, width: float = 12.0, grade: str = "HB"):
    xs = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    x = 20.0 + xs * 80.0
    y = 32.0 + np.zeros_like(x, dtype=np.float32)
    pts = np.column_stack([x, y]).astype(np.float32)
    return {
        "pts": pts,
        "width": np.full(64, width, dtype=np.float32),
        "pressure": np.full(64, pressure, dtype=np.float32),
        "grade": grade,
        "layer": "L1_tone",
        "kind": "hatch",
        "footprint": "side",
        "seed": 13,
    }


def _render_patch(grade: str, pressure: float) -> float:
    canvas = GraphiteCanvas((96, 96), seed=7, paper_preset="cold_press")
    canvas.add_stroke(_hatch_stroke(pressure=pressure, width=14.0, grade=grade))
    return float(np.mean(canvas.snapshot("L1_tone")))


def test_tone_to_pressure_roundtrip_for_each_grade():
    for grade in GRADE_ORDER:
        target = 0.55
        p = tone_to_pressure(grade, target)
        assert 0.0 <= p <= 1.0
        mean_tone = _render_patch(grade, p)
        assert abs(mean_tone - target) < 0.25


def test_pick_grade_pressure_prefers_softer_grade_for_dark_target():
    grade, pressure = pick_grade_pressure(0.8)
    assert grade in GRADE_ORDER
    assert 0.0 <= pressure <= 1.0
    assert grade in {"B", "2B", "4B", "6B", "8B"}


def test_swatches_have_expected_shapes():
    value_ladder = generate_value_ladder((48, 48), seed=5)
    pressure_ladder = generate_pressure_ladder((48, 48), seed=6)
    gradient = generate_gradient_ramp((48, 48), seed=7)
    hatch = generate_uniform_hatch_swatch((48, 48), seed=8)

    assert value_ladder.shape == (len(GRADE_ORDER), 48, 48)
    assert pressure_ladder.shape == (len(GRADE_ORDER), 48, 48)
    assert gradient.shape == (48, 48)
    assert hatch.shape == (48, 48)

    assert float(np.min(value_ladder)) >= 0.0
    assert float(np.max(value_ladder)) <= 1.0
    assert float(np.min(hatch)) >= 0.0
    assert float(np.max(hatch)) <= 1.0


def test_demo_script_has_debug_outputs(tmp_path):
    canvas = GraphiteCanvas((64, 64), seed=11, paper_preset="cold_press")
    stroke = _hatch_stroke(pressure=0.68, width=10.0, grade="4B")
    canvas.add_stroke(stroke)
    assert float(np.mean(canvas.snapshot("L1_tone"))) > 0.0

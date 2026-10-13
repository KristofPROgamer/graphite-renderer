from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from graphite.render.canvas import GraphiteCanvas
from graphite.render.graphite import GRADE_TABLE, grade_params

GRADE_ORDER = ("2H", "H", "HB", "B", "2B", "4B", "6B", "8B")


def _safe_pressure_value(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _make_hatch_stroke(*, pressure: float = 0.5, width: float = 10.0, grade: str = "HB", shape: tuple[int, int] = (96, 96), seed: int = 0) -> dict[str, Any]:
    h, w = shape
    xs = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    x = np.linspace(16.0, w - 16.0, len(xs), dtype=np.float32)
    y = (h * 0.5) + np.sin(xs * 8.0 + seed) * (min(h, w) * 0.18)
    pts = np.column_stack([x, y]).astype(np.float32)
    return {
        "pts": pts,
        "width": np.full(len(xs), width, dtype=np.float32),
        "pressure": np.full(len(xs), _safe_pressure_value(pressure), dtype=np.float32),
        "grade": grade,
        "layer": "L1_tone",
        "kind": "hatch",
        "footprint": "side",
        "seed": int(seed),
    }


def _render_grade_pressure(grade: str, pressure: float, *, shape: tuple[int, int] = (96, 96), seed: int = 0) -> float:
    canvas = GraphiteCanvas(shape, seed=seed, paper_preset="cold_press")
    canvas.add_stroke(_make_hatch_stroke(pressure=pressure, width=12.0, grade=grade, shape=shape, seed=seed))
    return float(np.mean(canvas.snapshot("L1_tone")))


def tone_to_pressure(grade: str, target_tone: float) -> float:
    """Estimate the pressure needed to hit a target graphite tone for a given grade."""
    if grade not in GRADE_TABLE:
        raise ValueError(f"Unknown graphite grade: {grade!r}")
    target = float(np.clip(target_tone, 0.0, 1.0))
    lo = 0.0
    hi = 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        value = _render_grade_pressure(grade, mid)
        if value < target:
            lo = mid
        else:
            hi = mid
    return _safe_pressure_value(0.5 * (lo + hi))


def pick_grade_pressure(target_tone: float) -> tuple[str, float]:
    """Pick a grade/pressure pair that targets the requested tone."""
    target = float(np.clip(target_tone, 0.0, 1.0))
    best_grade = "HB"
    best_pressure = 0.5
    best_error = float("inf")
    for grade in GRADE_ORDER:
        pressure = tone_to_pressure(grade, target)
        value = _render_grade_pressure(grade, pressure)
        err = abs(value - target)
        if err < best_error:
            best_grade = grade
            best_pressure = pressure
            best_error = err
    return best_grade, _safe_pressure_value(best_pressure)


def generate_value_ladder(shape: tuple[int, int], *, seed: int = 0) -> np.ndarray:
    """Generate a ladder of grade swatches from light to dark graphite."""
    h, w = shape
    ladder = np.zeros((len(GRADE_ORDER), h, w), dtype=np.float32)
    for i, grade in enumerate(GRADE_ORDER):
        p = tone_to_pressure(grade, 0.55)
        ladder[i] = np.full((h, w), _render_grade_pressure(grade, p, shape=shape, seed=seed + i), dtype=np.float32)
    return ladder


def generate_pressure_ladder(shape: tuple[int, int], *, seed: int = 0) -> np.ndarray:
    """Generate a pressure ladder for each graphite grade."""
    h, w = shape
    pressures = np.linspace(0.1, 1.0, len(GRADE_ORDER), dtype=np.float32)
    ladder = np.zeros((len(GRADE_ORDER), h, w), dtype=np.float32)
    for i, grade in enumerate(GRADE_ORDER):
        pressure = float(pressures[i])
        ladder[i] = np.full((h, w), _render_grade_pressure(grade, pressure, shape=shape, seed=seed + i), dtype=np.float32)
    return ladder


def generate_gradient_ramp(shape: tuple[int, int], *, seed: int = 0) -> np.ndarray:
    """Generate a monotone gradient ramp across tone values."""
    h, w = shape
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ramp = np.tile(ramp, (h, 1))
    return np.clip(ramp, 0.0, 1.0).astype(np.float32)


def generate_uniform_hatch_swatch(shape: tuple[int, int], *, seed: int = 0) -> np.ndarray:
    """Generate a simple hatch texture with uniform tonal response."""
    h, w = shape
    canvas = GraphiteCanvas(shape, seed=seed, paper_preset="cold_press")
    xs = np.linspace(0.0, 1.0, 48, dtype=np.float32)
    ys = np.linspace(0.1, 0.9, 48, dtype=np.float32)
    for i, y_frac in enumerate(ys):
        x0 = 8.0
        x1 = w - 8.0
        pts = np.column_stack([
            np.linspace(x0, x1, 36, dtype=np.float32),
            np.full(36, (y_frac * h), dtype=np.float32),
        ]).astype(np.float32)
        stroke = {
            "pts": pts,
            "width": np.full(36, 8.0, dtype=np.float32),
            "pressure": np.full(36, 0.45 + 0.3 * xs[i], dtype=np.float32),
            "grade": "HB",
            "layer": "L1_tone",
            "kind": "hatch",
            "footprint": "side",
            "seed": seed + i,
        }
        canvas.add_stroke(stroke)
    return canvas.snapshot("L1_tone")


def demo_module3(out_dir: str | Path = "out/debug") -> list[str]:
    """Create debug visualizations for the calibration module."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    value_ladder = generate_value_ladder((64, 64), seed=11)
    pressure_ladder = generate_pressure_ladder((64, 64), seed=12)
    gradient = generate_gradient_ramp((64, 64), seed=13)
    hatch = generate_uniform_hatch_swatch((64, 64), seed=14)

    saved: list[str] = []
    for name, arr in {
        "module3_value_ladder.png": value_ladder,
        "module3_pressure_ladder.png": pressure_ladder,
        "module3_gradient_ramp.png": gradient,
        "module3_hatch_swatch.png": hatch,
    }.items():
        img = np.clip(arr, 0.0, 1.0)
        if img.ndim == 3:
            if img.shape[0] == len(GRADE_ORDER):
                montage = img.copy()
                if montage.shape[0] == len(GRADE_ORDER):
                    montage = montage.reshape(len(GRADE_ORDER), -1)
                    img = montage
        Image.fromarray((img * 255).astype(np.uint8)).save(out / name)
        saved.append(str(out / name))
    return saved


__all__ = [
    "GRADE_ORDER",
    "tone_to_pressure",
    "pick_grade_pressure",
    "generate_value_ladder",
    "generate_pressure_ladder",
    "generate_gradient_ramp",
    "generate_uniform_hatch_swatch",
    "demo_module3",
]

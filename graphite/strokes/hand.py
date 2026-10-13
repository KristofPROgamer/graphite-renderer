from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Prefer the new `project_types.py` to avoid shadowing the stdlib `types` module.
_TYPES_SPEC = importlib.util.spec_from_file_location("graphite_project_types", ROOT / "project_types.py")
if _TYPES_SPEC is None or _TYPES_SPEC.loader is None:
    raise ImportError(f"Could not load project types module from {ROOT / 'project_types.py'}")
_project_types = importlib.util.module_from_spec(_TYPES_SPEC)
sys.modules[_TYPES_SPEC.name] = _project_types
_TYPES_SPEC.loader.exec_module(_project_types)
Stroke = _project_types.Stroke
from graphite.strokes.styles import StyleProfile, default_style_profiles


@dataclass
class NibScheduler:
    """Track nib wear and sharpen events along a stroke path."""

    initial_width: float = 5.0
    wear_rate: float = 0.12
    cumulative_distance: float = 0.0
    current_width: float = field(init=False)

    def __post_init__(self) -> None:
        self.current_width = float(self.initial_width)

    def advance(self, distance: float) -> float:
        """Advance the nib by a path distance and return the current width."""
        if distance <= 0:
            return float(self.current_width)
        self.cumulative_distance += float(distance)
        wear = self.wear_rate * float(distance)
        self.current_width = max(0.8, self.initial_width - 0.25 * np.sqrt(self.cumulative_distance * 0.18 + wear))
        if (self.cumulative_distance % 32.0) < distance:
            self.current_width = max(self.current_width * 0.9, 0.85)
        return float(self.current_width)


def _as_style(style: StyleProfile | dict[str, object] | None) -> StyleProfile:
    if style is None:
        return default_style_profiles()["generic"]
    if isinstance(style, StyleProfile):
        return style
    if isinstance(style, dict):
        return StyleProfile.from_dict(style)
    raise TypeError(f"Unsupported style type: {type(style).__name__}")


def _coerce_path(ideal_path: np.ndarray) -> np.ndarray:
    pts = np.asarray(ideal_path, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("ideal_path must have shape (N, 2) float32")
    if pts.shape[0] == 1:
        return np.vstack([pts, pts + 1.0]).astype(np.float32)
    return pts.astype(np.float32, copy=False)


def _polyline_lengths(path: np.ndarray) -> np.ndarray:
    if path.shape[0] < 2:
        return np.asarray([0.0], dtype=np.float32)
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(d, dtype=np.float32)])


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    safe = np.where(norms > 1e-8, norms, 1.0)
    return v / safe


def _compute_tangent(path: np.ndarray) -> np.ndarray:
    if path.shape[0] < 2:
        return np.zeros((1, 2), dtype=np.float32)
    vec = np.diff(path, axis=0)
    if vec.shape[0] == 0:
        return np.zeros((1, 2), dtype=np.float32)
    raw = np.zeros_like(path, dtype=np.float32)
    raw[1:] = vec
    raw[0] = vec[0]
    return _normalize(raw)


def _local_hand_noise(path: np.ndarray, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    if path.shape[0] == 0:
        return path
    tangent = _compute_tangent(path)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]).astype(np.float32)
    noise = rng.normal(0.0, amplitude, size=path.shape).astype(np.float32)
    noise = noise * np.clip(np.abs(normal), 0.0, 1.0)
    return path + noise


def make_stroke(ideal_path: np.ndarray, *, kind: str, style: StyleProfile | dict[str, object] | None, rng: np.random.Generator) -> Stroke:
    """Synthesize a single hand-drawn Stroke from an ideal polyline.

    The ideal path is the clean artist intention. The function adds tremor,
    short tapering, and a pressure profile to produce a realistic graphite stroke.
    """
    profile = _as_style(style)
    path = _coerce_path(ideal_path)
    if len(path) < 2:
        path = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    noise_amp = float(profile.jitter) * 0.35
    jittered = _local_hand_noise(path, rng, noise_amp)

    # build a mild curvature bias from the profile and path direction; this keeps the
    # drawing direction coherent while introducing a natural hand wobble.
    n = jittered.shape[0]
    if n > 2:
        tangent = _compute_tangent(jittered)
        curve_phase = np.linspace(0.0, np.pi * 2.0, n, dtype=np.float32)
        curve = np.sin(curve_phase + float(profile.curvature) * 8.0)
        perp = np.column_stack([-tangent[:, 1], tangent[:, 0]]).astype(np.float32)
        jittered = jittered + (curve[:, None] * perp) * (0.15 * float(profile.jitter))

    lengths = _polyline_lengths(jittered)
    if lengths[-1] <= 0:
        lengths = np.linspace(0.0, 1.0, n, dtype=np.float32)

    plateau = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ramp_in = np.clip(plateau * 3.5, 0.0, 1.0)
    taper_out = np.clip((1.0 - plateau) * 3.5, 0.0, 1.0)
    pressure = np.clip(float(profile.pressure) * np.minimum(ramp_in, taper_out), 0.0, 1.0).astype(np.float32)
    # `ramp_in`/`taper_out` are both 0 exactly at indices 0 and -1 (plateau's
    # own endpoints), for every n -- including n == 2, where those two points
    # ARE the whole stroke. Gating this floor behind `n > 2` left plain
    # 2-point strokes (e.g. a straight, RDP-simplified construction line from
    # graphite.strokes.construction.generate_construction_strokes) with
    # pressure=[0, 0], which renders as literally zero ink no matter the
    # style's nominal pressure (render_stroke's `dD = alpha * p**eta * ...`
    # is exactly 0 when p is 0). `n` is guaranteed >= 2 here (see
    # `_coerce_path` above), so this floor always applies.
    pressure[0] = 0.06
    pressure[-1] = 0.08

    width = np.full(n, float(profile.width), dtype=np.float32)
    width = width * (0.7 + 0.9 * pressure + 0.15 * rng.normal(0.0, 0.4, size=n))
    width = np.clip(width, 0.5, float(profile.width) * 2.6).astype(np.float32)

    layer = profile.layer_set[0] if profile.layer_set else "L1_tone"
    stroke_kind = kind or profile.kind
    footprint = "side" if stroke_kind in {"hatch", "scribble"} else "point"

    safe_seed = int(rng.integers(0, 2**31 - 1))
    return Stroke(
        pts=jittered.astype(np.float32),
        width=width.astype(np.float32),
        pressure=pressure.astype(np.float32),
        grade=profile.grade,
        layer=layer,
        kind=stroke_kind,
        footprint=footprint,
        seed=safe_seed,
    )


def break_into_hand_strokes(chain: np.ndarray, *, max_frac: float = 0.12) -> list[Stroke]:
    """Split a contour-like path into hand-length strokes.

    The natural hand-stroke length is bounded by a fraction of the chain's total
    length so contour and hatching strokes remain artistically plausible.
    """
    pts = np.asarray(chain, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("chain must have shape (N, 2)")
    if pts.shape[0] < 2:
        return []

    total_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    target = max(float(max_frac) * total_len, 12.0)
    style = default_style_profiles()["generic"]

    segments: list[np.ndarray] = []
    start = 0
    cumulative = 0.0
    for idx in range(1, pts.shape[0]):
        segment = float(np.linalg.norm(pts[idx] - pts[idx - 1]))
        cumulative += segment
        if cumulative >= target and idx < pts.shape[0] - 1:
            segments.append(pts[start:idx + 1])
            start = idx
            cumulative = 0.0
    if start < pts.shape[0]:
        segments.append(pts[start:])

    if not segments:
        segments = [pts]

    rng = np.random.default_rng(int(np.sum(np.abs(pts).astype(np.float32)) * 9713) % (2**31 - 1))
    strokes: list[Stroke] = []
    for seg in segments:
        if seg.shape[0] < 2:
            continue
        strokes.append(make_stroke(seg, kind="hatch", style=style, rng=rng))
    return strokes


def restate(stroke: Stroke, rng: np.random.Generator) -> Stroke:
    """Generate a second-pass restated stroke from the same path intent."""
    pts = np.asarray(stroke.pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("stroke.pts must have shape (N, 2)")
    perturbed = pts.copy()
    if pts.shape[0] > 2:
        tangent = _compute_tangent(pts)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]).astype(np.float32)
        noise = rng.normal(0.0, 0.35, size=pts.shape).astype(np.float32)
        perturbed = pts + noise * normal * 0.5
    pressure = np.clip(stroke.pressure * (0.85 + 0.25 * rng.random(pts.shape[0])), 0.0, 1.0).astype(np.float32)
    width = np.clip(stroke.width * (0.9 + 0.15 * rng.random(pts.shape[0])), 0.5, None).astype(np.float32)
    return Stroke(
        pts=perturbed.astype(np.float32),
        width=width,
        pressure=pressure,
        grade=stroke.grade,
        layer=stroke.layer,
        kind=stroke.kind,
        footprint=stroke.footprint,
        seed=int(rng.integers(0, 2**31 - 1)),
    )


__all__ = ["NibScheduler", "make_stroke", "break_into_hand_strokes", "restate"]

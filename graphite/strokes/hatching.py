from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles


def _coerce_field(name: str, array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _angle_to_unit(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)


def _sample_line_integral(T: np.ndarray, theta: np.ndarray, start: tuple[float, float], spacing: float, length: float) -> np.ndarray:
    H, W = T.shape
    points: list[tuple[float, float]] = []
    x, y = float(start[0]), float(start[1])
    dx, dy = float(np.cos(theta)), float(np.sin(theta))
    steps = max(2, int(math.ceil(length / max(spacing, 1e-6))))
    for _ in range(steps):
        points.append((x, y))
        x += dx * spacing
        y += dy * spacing
        if x < 0 or x >= W - 1 or y < 0 or y >= H - 1:
            break
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    return arr


def _generate_streamline(T: np.ndarray, theta: np.ndarray, x0: float, y0: float, spacing: float, *, rng: np.random.Generator, max_length: float = 120.0) -> np.ndarray:
    H, W = T.shape
    theta0 = float(theta[int(np.clip(y0, 0, H - 1)), int(np.clip(x0, 0, W - 1))])
    direction = np.array([np.cos(theta0), np.sin(theta0)], dtype=np.float32)
    jitter = rng.normal(0.0, 0.75, size=2).astype(np.float32)
    direction = direction + 0.08 * jitter
    direction /= np.linalg.norm(direction) + 1e-8

    pts: list[tuple[float, float]] = [(x0, y0)]
    x = float(x0)
    y = float(y0)
    total = 0.0
    while total < max_length:
        nx = x + direction[0] * spacing
        ny = y + direction[1] * spacing
        if nx < 0 or nx >= W - 1 or ny < 0 or ny >= H - 1:
            break
        pts.append((nx, ny))
        x, y = nx, ny
        total += spacing
        if total > max_length * 0.6:
            # mild curvature and wobble toward the local hatching direction
            direction = direction + 0.06 * np.array([np.cos(theta0 + 0.2 * np.sin(total / 12.0)), np.sin(theta0 + 0.2 * np.cos(total / 10.0))], dtype=np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
    if len(pts) < 2:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _layer_spacing(T: np.ndarray, layer_index: int, base_spacing: float) -> float:
    # Lower tone receives denser hatching; darker regions get tighter spacing.
    level = 0.25 + layer_index * 0.18
    tone_bias = 1.0 - np.clip(T.mean(), 0.0, 1.0)
    return max(1.2, base_spacing * (1.2 + level * 0.4 + tone_bias * 0.8))


def _residual_error(T: np.ndarray, rendered: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(T, dtype=np.float32) - np.asarray(rendered, dtype=np.float32)
    return gaussian_filter(np.asarray(T, dtype=np.float32), sigma=sigma, mode="nearest", truncate=3.0) - gaussian_filter(
        np.asarray(rendered, dtype=np.float32),
        sigma=sigma,
        mode="nearest",
        truncate=3.0,
    )


def residual_loop(T: np.ndarray, rendered: np.ndarray, *, theta: np.ndarray | None = None, W: np.ndarray | None = None) -> np.ndarray:
    """Compute a coarse-to-fine residual map for a hatching pass.

    The residual is the smoothed target tone minus the current render, which can be
    used to place more strokes only where the drawing remains too light.
    """
    T = np.asarray(T, dtype=np.float32)
    rendered = np.asarray(rendered, dtype=np.float32)
    if T.shape != rendered.shape:
        raise ValueError(f"T and rendered must share shape, got {T.shape} and {rendered.shape}")
    err = _residual_error(T, rendered, sigma=1.5)
    if W is not None:
        W = np.asarray(W, dtype=np.float32)
        if W.shape != T.shape:
            raise ValueError(f"W must have shape {T.shape}, got {W.shape}")
        err = err * np.clip(W, 0.0, 1.0)
    return np.clip(err, 0.0, None).astype(np.float32)


def generate_hatching_strokes(
    T: np.ndarray,
    theta: np.ndarray,
    *,
    W: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    max_strokes: int = 4000,
) -> list:
    """Generate hatching strokes for a tone map using local streamline placement.

    The routine samples a sparse set of streamlines aligned with the local axial
    orientation ``theta`` and converts them into real ``Stroke`` instances. The
    produced strokes carry L1 tone layer metadata and use the default style profiles.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = np.asarray(T, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    if T.ndim != 2:
        raise ValueError(f"T must be a 2D array, got shape {T.shape}")
    if theta.shape != T.shape:
        raise ValueError(f"theta must have shape {T.shape}, got {theta.shape}")
    if W is None:
        W = np.ones_like(T, dtype=np.float32)
    else:
        W = np.asarray(W, dtype=np.float32)
        if W.shape != T.shape:
            raise ValueError(f"W must have shape {T.shape}, got {W.shape}")

    styles = default_style_profiles()
    style = styles["generic"]
    strokes: list = []
    H, W2 = T.shape

    # Seed a coarse set of streamlines that are denser in dark regions.
    spacing = 10.0
    base_density = 0.75
    n_candidates = min(max_strokes, max(12, int(H * W2 * base_density / 60)))
    for _ in range(n_candidates):
        x = float(rng.uniform(0, W2 - 1))
        y = float(rng.uniform(0, H - 1))
        local_T = float(T[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W2 - 1))])
        local_W = float(W[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W2 - 1))])
        if local_W <= 0.0 or local_T >= 0.82:
            continue
        if local_T > 0.9:
            continue
        local_theta = float(theta[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W2 - 1))])
        path = _generate_streamline(T, theta, x, y, spacing, rng=rng, max_length=90.0)
        if path.shape[0] < 2:
            continue
        # T is a darkness target (1 = near-black, needs heavy deposit; 0 = paper
        # white, needs none -- see graphite.plan.tone.compute_tone_target and
        # graphite.render.calibration.tone_to_pressure, which searches for the
        # pressure whose rendered deposit matches target_tone directly). So
        # pressure/width must scale UP with local_T, not with (1 - local_T).
        pressure = np.clip(style.pressure * (0.65 + local_T * 0.75), 0.0, 1.0)
        width = np.full(path.shape[0], style.width * (0.8 + local_T * 0.6), dtype=np.float32)
        s = make_stroke(
            path,
            kind="hatch",
            style={
                "name": "hatch",
                "kind": "hatch",
                "grade": style.grade,
                "width": float(style.width),
                "pressure": float(pressure),
                "jitter": style.jitter,
                "curvature": style.curvature,
                "spacing_multiplier": style.spacing_multiplier,
                "blend_amount": style.blend_amount,
                "layer_set": ("L1_tone",),
            },
            rng=rng,
        )
        s.width = width.astype(np.float32)
        s.pressure = np.clip(s.pressure * (0.7 + 0.7 * local_T), 0.0, 1.0).astype(np.float32)
        s.layer = "L1_tone"
        s.validate()
        strokes.append(s)

    return strokes


__all__ = ["generate_hatching_strokes", "residual_loop"]

from __future__ import annotations

from typing import Any

import numpy as np

from graphite.core.geometry import resample_polyline

try:
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args else decorator

_GRADE_TABLE = {
    # alpha/kappa/width_scale recalibrated for the physically-correct
    # Gaussian-footprint + smoothstep-contact stamp model (see
    # _stamp_patch/render_stroke below). The previous values were tuned
    # against the old ad hoc box-fill deposit model that Step 3 replaced,
    # and left every grade's achievable single-stroke tone far below what
    # calibration.tone_to_pressure needs to hit (e.g. 2H maxed out at a
    # mean tone of ~0.008 at full pressure instead of tracking target
    # tones around 0.5). eta and dark_floor are unchanged: eta only feeds
    # the alpha*p**eta term (independent of the box-fill bug) and
    # dark_floor is unused dead metadata (grep confirms no reader).
    "2H": {"grade": "2H", "alpha": 0.2500, "kappa": 1.000, "eta": 1.1, "width_scale": 2.8473, "dark_floor": 0.028},
    "H": {"grade": "H", "alpha": 0.3203, "kappa": 1.071, "eta": 1.2, "width_scale": 2.5665, "dark_floor": 0.042},
    "HB": {"grade": "HB", "alpha": 0.3730, "kappa": 1.143, "eta": 1.35, "width_scale": 2.5249, "dark_floor": 0.0595},
    "B": {"grade": "B", "alpha": 0.4503, "kappa": 1.214, "eta": 1.45, "width_scale": 2.3854, "dark_floor": 0.077},
    "2B": {"grade": "2B", "alpha": 0.5416, "kappa": 1.286, "eta": 1.6, "width_scale": 2.3384, "dark_floor": 0.0945},
    "4B": {"grade": "4B", "alpha": 0.6470, "kappa": 1.357, "eta": 1.75, "width_scale": 2.2631, "dark_floor": 0.1155},
    "6B": {"grade": "6B", "alpha": 0.7770, "kappa": 1.429, "eta": 1.9, "width_scale": 2.2466, "dark_floor": 0.14},
    "8B": {"grade": "8B", "alpha": 0.9000, "kappa": 1.500, "eta": 2.05, "width_scale": 2.2629, "dark_floor": 0.1645},
}
GRADE_TABLE = {k: dict(v) for k, v in _GRADE_TABLE.items()}


def grade_params(grade: str) -> dict[str, float | str]:
    """Return grade-specific rendering parameters for the graphite deposit model."""
    if grade not in GRADE_TABLE:
        raise ValueError(f"Unknown graphite grade: {grade!r}")
    return dict(GRADE_TABLE[grade])


@njit(cache=True)
def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    denom = edge1 - edge0
    if denom <= 1e-12:
        return 1.0 if x >= edge1 else 0.0
    t = (x - edge0) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def _elliptical_footprint(dx: float, dy: float, r: float) -> float:
    # Radial Gaussian nib footprint centered on the stamp, sigma ~= r.
    sigma = max(r, 1e-6)
    dist2 = dx * dx + dy * dy
    return np.exp(-0.5 * dist2 / (sigma * sigma))


@njit(cache=True)
def _stamp_patch(D: np.ndarray, H: np.ndarray, x0: int, y0: int, x1: int, y1: int, cx: float, cy: float, r: float, p: float, alpha: float, kappa: float, eta: float, eps: float) -> None:
    # reach: how far the nib presses into the paper tooth at this pressure.
    # Increasing pressure shrinks `reach`, which *expands* the set of paper
    # heights h that satisfy h >= reach -- i.e. higher pressure makes contact
    # with more of the paper surface (not just the tallest tooth peaks),
    # matching how a real pencil flattens into paper under more force.
    reach = 1.0 - kappa * p
    for y in range(y0, y1):
        dy = y - cy
        for x in range(x0, x1):
            dx = x - cx
            h = H[y, x]
            contact = _smoothstep(reach - eps, reach + eps, h)
            fp = _elliptical_footprint(dx, dy, r)
            dD = alpha * (p ** eta) * fp * contact
            if dD <= 0.0:
                continue
            # Single saturating deposit -- no double-application.
            D[y, x] = 1.0 - (1.0 - D[y, x]) * (1.0 - dD)


def _subsample_step(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    return float(np.median(lengths)) if lengths.size else 1.0


def _resample_stroke_values(pts: np.ndarray, width: np.ndarray, pressure: np.ndarray, *, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a stroke's path at fixed arc-length spacing, interpolating
    ``width``/``pressure`` onto the same arc-length positions as the
    resampled points so all three stay index-aligned.

    Positions are resampled via ``graphite.core.geometry.resample_polyline``;
    ``width``/``pressure`` are linearly interpolated along the same
    cumulative arc-length parametrization that function uses internally.
    """
    pts64 = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts64, axis=0), axis=1) if pts64.shape[0] > 1 else np.zeros(0)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumlen[-1])
    rp = resample_polyline(pts64, spacing=spacing)
    n = rp.shape[0]
    if total <= 1e-9:
        rw = np.full(n, float(width[0]) if width.size else 1.0, dtype=np.float32)
        rpres = np.full(n, float(pressure[0]) if pressure.size else 0.5, dtype=np.float32)
        return rp, rw, rpres
    targets = np.linspace(0.0, total, n)
    rw = np.interp(targets, cumlen, width).astype(np.float32)
    rpres = np.interp(targets, cumlen, pressure).astype(np.float32)
    return rp, rw, rpres


def _stroke_field(stroke: Any, name: str, default: Any = None) -> Any:
    """Read a field from a stroke, whether it's a dict or a ``Stroke`` dataclass.

    Strokes flow through this module from two different sources: hand-built
    dicts (tests, calibration swatches, demos) and ``Stroke`` dataclass
    instances produced by ``graphite.strokes.hand.make_stroke`` (used by every
    real stroke-generation module: hatching, contours, construction, stipple,
    textures). Both must be renderable here.
    """
    if isinstance(stroke, dict):
        return stroke.get(name, default)
    return getattr(stroke, name, default)


def render_stroke(stroke: Any, canvas: Any, *, layer: str = "L1_tone") -> Any:
    """Render a stroke (dict or ``Stroke`` dataclass) onto a GraphiteCanvas in the named layer.

    Physical deposit model: the stroke path is resampled to a fixed
    arc-length spacing (~0.3-0.5x the nib radius) and, at each resampled
    point, a single soft circular stamp is deposited -- a radial Gaussian
    footprint (sigma ~= nib radius) gated by a smoothstep "contact" term that
    compares how far the nib reaches into the paper tooth (grade/pressure
    dependent) against the local paper heightfield. Deposits accumulate via
    ``D = 1 - (1-D)*(1-dD)`` (one deposit per point) so ink saturates
    smoothly toward 1.0 instead of blowing out from overlapping stamps.
    """
    pts = np.asarray(_stroke_field(stroke, "pts"), dtype=np.float32)
    width = np.asarray(_stroke_field(stroke, "width"), dtype=np.float32)
    pressure = np.asarray(_stroke_field(stroke, "pressure"), dtype=np.float32)
    grade = str(_stroke_field(stroke, "grade", "HB"))
    params = grade_params(grade)
    if pts.shape[0] == 0:
        return canvas
    if pts.shape[0] == 1:
        pts = np.vstack([pts, pts])
    if width.shape[0] != pts.shape[0]:
        width = np.full(pts.shape[0], float(np.median(width)) if width.size else 6.0, dtype=np.float32)
    if pressure.shape[0] != pts.shape[0]:
        pressure = np.full(pts.shape[0], float(np.mean(pressure)) if pressure.size else 0.5, dtype=np.float32)

    D = canvas.layers[layer]
    paper = canvas.paper_heightfield
    width_scale = float(params["width_scale"])
    alpha = float(params["alpha"])
    kappa = float(params["kappa"])
    eta = float(params["eta"])

    # Representative nib radius for this stroke, used only to pick the
    # resampling spacing (resample_polyline needs one fixed spacing value).
    r_repr = max(0.5, 0.5 * float(np.median(width)) * width_scale)
    spacing = max(0.5, 0.4 * r_repr)

    rpts, rwidth, rpressure = _resample_stroke_values(pts, width, pressure, spacing=spacing)

    h, w = D.shape
    eps = 0.02  # smoothstep half-width for the contact transition
    for i in range(rpts.shape[0]):
        cx = float(rpts[i, 0])
        cy = float(rpts[i, 1])
        p = float(np.clip(rpressure[i], 0.0, 1.0))
        r = max(0.5, 0.5 * float(rwidth[i]) * width_scale)
        patch_r = max(1.0, 3.0 * r)  # ~3 sigma covers the Gaussian tail
        x0 = max(0, int(np.floor(cx - patch_r)))
        x1 = min(w, int(np.ceil(cx + patch_r)) + 1)
        y0 = max(0, int(np.floor(cy - patch_r)))
        y1 = min(h, int(np.ceil(cy + patch_r)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        _stamp_patch(D, paper, x0, y0, x1, y1, cx, cy, r, p, alpha, kappa, eta, eps)
    return canvas

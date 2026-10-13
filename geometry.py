"""Geometry helpers shared by contour tracing, stroke construction, and rendering.

Conventions follow the Global Context: points are ``(x, y)`` float, x right,
y down, in working-resolution pixels. Orientation ``theta`` is *axial*
(``[0, pi)``, a line has no direction) and must always be averaged/blended via
:func:`average_orientation` or the tensor helpers below -- never by naively
averaging angle values, which breaks across the 0/pi wraparound.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

_EPS = 1e-12

# ---------------------------------------------------------------------------
# Polyline resampling / simplification / smoothing
# ---------------------------------------------------------------------------


def polyline_length(pts: np.ndarray) -> float:
    """Total arc length of a polyline ``(N, 2)``."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_polyline(
    pts: np.ndarray,
    *,
    spacing: Optional[float] = None,
    n_points: Optional[int] = None,
) -> np.ndarray:
    """Resample a polyline to uniform arc-length spacing.

    Exactly one of ``spacing`` or ``n_points`` must be given. Degenerate
    (single-point or zero-length) inputs are returned as ``n_points`` (or 2)
    repeats of that point rather than raising, since callers (e.g. dot
    stamping) may legitimately pass zero-length paths.

    Args:
        pts: ``(N, 2)`` polyline, ``N >= 1``.
        spacing: desired arc-length spacing between output points.
        n_points: exact number of output points (``>= 2``).

    Returns:
        ``(M, 2)`` float32 resampled polyline.
    """
    if (spacing is None) == (n_points is None):
        raise ValueError("resample_polyline: pass exactly one of spacing or n_points")
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[-1] != 2 or pts.shape[0] < 1:
        raise ValueError(f"resample_polyline: pts must be (N,2) with N>=1, got shape {pts.shape}")

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1) if pts.shape[0] > 1 else np.zeros(0)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumlen[-1])

    if total <= _EPS:
        n = n_points if n_points is not None else 2
        return np.repeat(pts[:1], max(n, 1), axis=0).astype(np.float32)

    if spacing is not None:
        if spacing <= 0:
            raise ValueError(f"resample_polyline: spacing must be > 0, got {spacing}")
        n = max(2, int(round(total / spacing)) + 1)
    else:
        n = n_points
        if n < 2:
            raise ValueError(f"resample_polyline: n_points must be >= 2, got {n}")

    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, cumlen, pts[:, 0])
    y = np.interp(targets, cumlen, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def smooth_polyline(pts: np.ndarray, *, sigma: float = 1.0, closed: bool = False) -> np.ndarray:
    """Gaussian-smooth a polyline's x/y coordinates independently.

    Args:
        pts: ``(N, 2)`` polyline.
        sigma: Gaussian standard deviation, in samples.
        closed: if True, wrap at the ends (for closed contours) instead of
            using nearest-edge reflection.

    Returns:
        ``(N, 2)`` float32 smoothed polyline.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3 or sigma <= 0:
        return pts.astype(np.float32)
    mode = "wrap" if closed else "nearest"
    x = gaussian_filter1d(pts[:, 0], sigma=sigma, mode=mode)
    y = gaussian_filter1d(pts[:, 1], sigma=sigma, mode=mode)
    return np.stack([x, y], axis=1).astype(np.float32)


def douglas_peucker(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify a polyline with the Ramer-Douglas-Peucker algorithm.

    Args:
        pts: ``(N, 2)`` polyline.
        epsilon: maximum perpendicular deviation allowed when dropping a point.

    Returns:
        ``(M, 2)`` float32 simplified polyline, ``M <= N``, endpoints kept.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return pts.astype(np.float32)
    if epsilon < 0:
        raise ValueError(f"douglas_peucker: epsilon must be >= 0, got {epsilon}")

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        seg = p1 - p0
        seg_len = np.hypot(*seg)
        idx = np.arange(i0 + 1, i1)
        if seg_len < _EPS:
            dist = np.linalg.norm(pts[idx] - p0, axis=1)
        else:
            # perpendicular distance from each point to the line p0-p1
            cross = seg[0] * (p0[1] - pts[idx, 1]) - seg[1] * (p0[0] - pts[idx, 0])
            dist = np.abs(cross) / seg_len
        if dist.size == 0:
            continue
        k = int(np.argmax(dist))
        if dist[k] > epsilon:
            far = i0 + 1 + k
            keep[far] = True
            stack.append((i0, far))
            stack.append((far, i1))
    return pts[keep].astype(np.float32)


# ---------------------------------------------------------------------------
# Cubic Bezier evaluation
# ---------------------------------------------------------------------------


def bezier_eval(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Bezier curve.

    Args:
        ctrl: ``(4, 2)`` control points ``p0, p1, p2, p3``.
        t: scalar or ``(M,)`` parameter values, typically in ``[0, 1]``.

    Returns:
        ``(M, 2)`` float32 points on the curve (or ``(2,)`` if ``t`` was scalar).
    """
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if ctrl.shape != (4, 2):
        raise ValueError(f"bezier_eval: ctrl must have shape (4,2), got {ctrl.shape}")
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64)).reshape(-1, 1)
    p0, p1, p2, p3 = ctrl
    mt = 1.0 - t_arr
    pts = (mt**3) * p0 + 3 * (mt**2) * t_arr * p1 + 3 * mt * (t_arr**2) * p2 + (t_arr**3) * p3
    pts = pts.astype(np.float32)
    return pts[0] if np.isscalar(t) else pts


def bezier_derivative(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """First derivative (tangent vector, not normalized) of a cubic Bezier."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64)).reshape(-1, 1)
    p0, p1, p2, p3 = ctrl
    mt = 1.0 - t_arr
    d = 3 * (mt**2) * (p1 - p0) + 6 * mt * t_arr * (p2 - p1) + 3 * (t_arr**2) * (p3 - p2)
    d = d.astype(np.float32)
    return d[0] if np.isscalar(t) else d


def bezier_second_derivative(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Second derivative of a cubic Bezier."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64)).reshape(-1, 1)
    p0, p1, p2, p3 = ctrl
    mt = 1.0 - t_arr
    d2 = 6 * mt * (p2 - 2 * p1 + p0) + 6 * t_arr * (p3 - 2 * p2 + p1)
    d2 = d2.astype(np.float32)
    return d2[0] if np.isscalar(t) else d2


def bezier_curvature(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Signed curvature ``kappa = (x'y'' - y'x'') / (x'^2+y'^2)^1.5`` along a cubic Bezier."""
    d1 = np.atleast_2d(bezier_derivative(ctrl, t).astype(np.float64))
    d2 = np.atleast_2d(bezier_second_derivative(ctrl, t).astype(np.float64))
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    speed = np.linalg.norm(d1, axis=1)
    kappa = cross / (speed**3 + _EPS)
    kappa = kappa.astype(np.float32)
    return float(kappa[0]) if np.isscalar(t) else kappa


# ---------------------------------------------------------------------------
# Tangent / normal helpers
# ---------------------------------------------------------------------------


def polyline_tangents(pts: np.ndarray, *, closed: bool = False) -> np.ndarray:
    """Per-point unit tangent vectors of a polyline via central differences.

    Args:
        pts: ``(N, 2)`` polyline, ``N >= 2``.
        closed: whether to treat the polyline as a closed loop at the ends.

    Returns:
        ``(N, 2)`` float32 unit tangent vectors.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("polyline_tangents: need at least 2 points")
    if closed:
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
    else:
        prev = np.vstack([pts[:1], pts[:-1]])
        nxt = np.vstack([pts[1:], pts[-1:]])
    d = nxt - prev
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm = np.where(norm < _EPS, 1.0, norm)
    return (d / norm).astype(np.float32)


def theta_to_tangent_normal(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert axial orientation(s) to unit tangent/normal vectors.

    Args:
        theta: ``(...,)`` axial angle(s) in radians.

    Returns:
        ``(tangent, normal)``, each ``(..., 2)`` float32; ``normal`` is
        ``tangent`` rotated +90 degrees (x right, y down convention).
    """
    theta = np.asarray(theta, dtype=np.float64)
    tangent = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)
    return tangent, normal


# ---------------------------------------------------------------------------
# Axial orientation <-> structure tensor, with wraparound-safe averaging
# ---------------------------------------------------------------------------


def orientation_to_vector(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode axial angle(s) as a doubled-angle unit vector ``(cos 2theta, sin 2theta)``.

    Doubling the angle maps the axial domain ``[0, pi)`` (period pi, no
    direction) onto a full turn (period ``2*pi``), which is what makes linear
    (vector) averaging of the result correct for axial data -- see
    :func:`average_orientation`.
    """
    theta = np.asarray(theta, dtype=np.float64)
    return np.cos(2 * theta), np.sin(2 * theta)


def vector_to_orientation(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`orientation_to_vector`.

    Returns:
        ``(theta, coherence)``: ``theta`` in ``[0, pi)``, ``coherence`` = the
        vector's magnitude (0 = fully incoherent/canceled, 1 = perfectly
        aligned unit vectors summed).
    """
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    theta = 0.5 * np.mod(np.arctan2(vy, vx), 2 * np.pi)
    coherence = np.hypot(vx, vy)
    return theta.astype(np.float32), coherence.astype(np.float32)


def average_orientation(
    thetas: np.ndarray, weights: Optional[np.ndarray] = None
) -> tuple[float, float]:
    """Weighted average of axial angles, safe across the 0/pi wraparound.

    Implemented by averaging the doubled-angle unit vectors (see
    :func:`orientation_to_vector`) rather than the angles themselves, so e.g.
    averaging 179 degrees and 1 degree (nearly opposite as raw numbers, but
    the *same* axial line up to the pi-periodicity) gives ~0/180 degrees, not
    the ~90 degrees a naive mean would produce.

    Args:
        thetas: ``(N,)`` axial angles in radians.
        weights: optional ``(N,)`` non-negative weights; uniform if omitted.

    Returns:
        ``(theta, coherence)`` as Python floats.
    """
    thetas = np.asarray(thetas, dtype=np.float64).ravel()
    if weights is None:
        weights = np.ones_like(thetas)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
    total_w = np.sum(weights)
    if total_w <= _EPS:
        return 0.0, 0.0
    vx, vy = orientation_to_vector(thetas)
    vx_avg = float(np.sum(weights * vx) / total_w)
    vy_avg = float(np.sum(weights * vy) / total_w)
    theta, coherence = vector_to_orientation(vx_avg, vy_avg)
    return float(theta), float(coherence)


def structure_tensor_field(
    img_gray: np.ndarray, *, sigma_grad: float = 1.0, sigma_tensor: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel structure tensor field of a grayscale image.

    Args:
        img_gray: ``(H, W)`` image.
        sigma_grad: Gaussian derivative scale for the initial gradient.
        sigma_tensor: Gaussian blur scale applied to the outer-product tensor
            components (the averaging window).

    Returns:
        ``(Jxx, Jxy, Jyy)``, each ``(H, W)`` float32: the blurred structure
        tensor components. Pass to :func:`tensor_to_orientation` to get an
        orientation field + coherence.
    """
    img_gray = np.asarray(img_gray, dtype=np.float64)
    gx = gaussian_filter(img_gray, sigma=sigma_grad, order=(0, 1))
    gy = gaussian_filter(img_gray, sigma=sigma_grad, order=(1, 0))
    Jxx = gaussian_filter(gx * gx, sigma=sigma_tensor)
    Jxy = gaussian_filter(gx * gy, sigma=sigma_tensor)
    Jyy = gaussian_filter(gy * gy, sigma=sigma_tensor)
    return Jxx.astype(np.float32), Jxy.astype(np.float32), Jyy.astype(np.float32)


def tensor_to_orientation(
    Jxx: np.ndarray, Jxy: np.ndarray, Jyy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Dominant axial orientation + coherence of a structure tensor field.

    Args:
        Jxx, Jxy, Jyy: structure tensor components, e.g. from
            :func:`structure_tensor_field`.

    Returns:
        ``(theta, coherence)``, each same shape, ``theta`` in ``[0, pi)`` and
        ``coherence`` in ``[0, 1]`` (``(lambda1-lambda2)/(lambda1+lambda2)``).
    """
    Jxx = np.asarray(Jxx, dtype=np.float64)
    Jxy = np.asarray(Jxy, dtype=np.float64)
    Jyy = np.asarray(Jyy, dtype=np.float64)
    theta = 0.5 * np.mod(np.arctan2(2 * Jxy, Jxx - Jyy), 2 * np.pi)
    denom = Jxx + Jyy
    num = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy**2)
    coherence = np.where(denom > _EPS, num / np.maximum(denom, _EPS), 0.0)
    coherence = np.clip(coherence, 0.0, 1.0)
    return theta.astype(np.float32), coherence.astype(np.float32)


def average_orientation_field(
    theta: np.ndarray, coherence: np.ndarray, *, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Spatially blur a ``(theta, coherence)`` orientation field, wraparound-safe.

    Equivalent to converting to doubled-angle vectors, Gaussian-blurring each
    component, and converting back -- i.e. :func:`average_orientation` applied
    per-pixel over a local Gaussian window.

    Args:
        theta: ``(H, W)`` axial angle field, radians.
        coherence: ``(H, W)`` weight/coherence in ``[0, 1]``.
        sigma: Gaussian blur standard deviation, in pixels.

    Returns:
        ``(theta_smoothed, coherence_smoothed)``, each ``(H, W)`` float32.
    """
    vx, vy = orientation_to_vector(theta)
    vx = vx * coherence
    vy = vy * coherence
    vx_b = gaussian_filter(vx, sigma=sigma)
    vy_b = gaussian_filter(vy, sigma=sigma)
    return vector_to_orientation(vx_b, vy_b)


# ---------------------------------------------------------------------------
# Bilinear sampling
# ---------------------------------------------------------------------------


def bilinear_sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinearly sample an image (or mask) at float ``(x, y)`` points.

    Coordinates outside the image are clamped to the nearest edge (a bool
    mask sampled this way yields a float coverage-like value in ``[0, 1]``
    at boundaries between True/False).

    Args:
        img: ``(H, W)`` or ``(H, W, C)`` array (any real dtype).
        pts: ``(N, 2)`` float ``(x, y)`` coordinates.

    Returns:
        ``(N,)`` (for 2D ``img``) or ``(N, C)`` (for 3D ``img``) float32 samples.
    """
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"bilinear_sample: img must be (H,W) or (H,W,C), got shape {img.shape}")
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[-1] != 2:
        raise ValueError(f"bilinear_sample: pts must be (N,2), got shape {pts.shape}")

    H, W = img.shape[:2]
    x = np.clip(pts[:, 0], 0.0, W - 1)
    y = np.clip(pts[:, 1], 0.0, H - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = (x - x0).astype(np.float64)
    wy = (y - y0).astype(np.float64)

    img_f = img.astype(np.float64)
    Ia = img_f[y0, x0]
    Ib = img_f[y0, x1]
    Ic = img_f[y1, x0]
    Id = img_f[y1, x1]

    if img.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]

    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)

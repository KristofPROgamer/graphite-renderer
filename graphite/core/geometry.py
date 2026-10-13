from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

_EPS = 1e-12


def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_polyline(pts: np.ndarray, *, spacing: Optional[float] = None, n_points: Optional[int] = None) -> np.ndarray:
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
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3 or sigma <= 0:
        return pts.astype(np.float32)
    mode = "wrap" if closed else "nearest"
    x = gaussian_filter1d(pts[:, 0], sigma=sigma, mode=mode)
    y = gaussian_filter1d(pts[:, 1], sigma=sigma, mode=mode)
    return np.stack([x, y], axis=1).astype(np.float32)


def douglas_peucker(pts: np.ndarray, epsilon: float) -> np.ndarray:
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


def bezier_eval(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
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
    ctrl = np.asarray(ctrl, dtype=np.float64)
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64)).reshape(-1, 1)
    p0, p1, p2, p3 = ctrl
    mt = 1.0 - t_arr
    d = 3 * (mt**2) * (p1 - p0) + 6 * mt * t_arr * (p2 - p1) + 3 * (t_arr**2) * (p3 - p2)
    d = d.astype(np.float32)
    return d[0] if np.isscalar(t) else d


def bezier_second_derivative(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    ctrl = np.asarray(ctrl, dtype=np.float64)
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64)).reshape(-1, 1)
    p0, p1, p2, p3 = ctrl
    mt = 1.0 - t_arr
    d2 = 6 * mt * (p2 - 2 * p1 + p0) + 6 * t_arr * (p3 - 2 * p2 + p1)
    d2 = d2.astype(np.float32)
    return d2[0] if np.isscalar(t) else d2


def bezier_curvature(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    d1 = np.atleast_2d(bezier_derivative(ctrl, t).astype(np.float64))
    d2 = np.atleast_2d(bezier_second_derivative(ctrl, t).astype(np.float64))
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    speed = np.linalg.norm(d1, axis=1)
    kappa = cross / (speed**3 + _EPS)
    return kappa.astype(np.float32)


def polyline_tangents(pts: np.ndarray, *, closed: bool = False) -> np.ndarray:
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
    theta = np.asarray(theta, dtype=np.float64)
    tangent = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)
    return tangent, normal


def orientation_to_vector(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    return np.cos(2 * theta), np.sin(2 * theta)


def vector_to_orientation(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    theta = 0.5 * np.mod(np.arctan2(vy, vx), 2 * np.pi)
    coherence = np.hypot(vx, vy)
    return theta.astype(np.float32), coherence.astype(np.float32)


def average_orientation(thetas: np.ndarray, weights: Optional[np.ndarray] = None) -> tuple[float, float]:
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


def structure_tensor_field(img_gray: np.ndarray, *, sigma_grad: float = 1.0, sigma_tensor: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_gray = np.asarray(img_gray, dtype=np.float64)
    gx = gaussian_filter(img_gray, sigma=sigma_grad, order=(0, 1))
    gy = gaussian_filter(img_gray, sigma=sigma_grad, order=(1, 0))
    Jxx = gaussian_filter(gx * gx, sigma=sigma_tensor)
    Jxy = gaussian_filter(gx * gy, sigma=sigma_tensor)
    Jyy = gaussian_filter(gy * gy, sigma=sigma_tensor)
    return Jxx.astype(np.float32), Jxy.astype(np.float32), Jyy.astype(np.float32)


def tensor_to_orientation(Jxx: np.ndarray, Jxy: np.ndarray, Jyy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Jxx = np.asarray(Jxx, dtype=np.float64)
    Jxy = np.asarray(Jxy, dtype=np.float64)
    Jyy = np.asarray(Jyy, dtype=np.float64)
    theta = 0.5 * np.mod(np.arctan2(2 * Jxy, Jxx - Jyy), 2 * np.pi)
    denom = Jxx + Jyy
    num = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy**2)
    coherence = np.where(denom > _EPS, num / np.maximum(denom, _EPS), 0.0)
    coherence = np.clip(coherence, 0.0, 1.0)
    return theta.astype(np.float32), coherence.astype(np.float32)


def average_orientation_field(theta: np.ndarray, coherence: np.ndarray, *, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    vx, vy = orientation_to_vector(theta)
    vx = vx * coherence
    vy = vy * coherence
    vx_b = gaussian_filter(vx, sigma=sigma)
    vy_b = gaussian_filter(vy, sigma=sigma)
    return vector_to_orientation(vx_b, vy_b)


def bilinear_sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
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

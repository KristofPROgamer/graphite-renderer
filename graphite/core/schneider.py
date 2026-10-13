"""Cubic Bezier fitting (Schneider algorithm, simplified).

Provides a lightweight recursive fitter that approximates a polyline with
one or more cubic Beziers within a user tolerance. Output beziers are
arrays of shape (4,2) control points in working-resolution pixels.

This implementation is intentionally self-contained and avoids heavy
dependencies so tests and the plan builder remain deterministic.
"""
from __future__ import annotations

import numpy as np
from typing import List


def _chord_length_parameterize(pts: np.ndarray) -> np.ndarray:
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    u = np.zeros(len(pts), dtype=np.float32)
    if d.sum() == 0:
        return u
    u[1:] = np.cumsum(d) / d.sum()
    return u


def _bezier_point(ctrl: np.ndarray, t: float) -> np.ndarray:
    # cubic bezier eval (4,2)
    t2 = t * t
    t3 = t2 * t
    u = 1.0 - t
    u2 = u * u
    u3 = u2 * u
    return u3 * ctrl[0] + 3 * u2 * t * ctrl[1] + 3 * u * t2 * ctrl[2] + t3 * ctrl[3]


def _least_squares_fit(pts: np.ndarray, u: np.ndarray, left_tan: np.ndarray, right_tan: np.ndarray):
    # Solve for inner control points P1 and P2 (see Schneider)
    n = len(pts)
    A = np.zeros((n, 2, 2), dtype=np.float64)
    C = np.zeros((2, 2), dtype=np.float64)
    X = np.zeros(2, dtype=np.float64)
    P0 = pts[0]
    P3 = pts[-1]
    for i in range(n):
        ui = u[i]
        b0 = (1 - ui) ** 3
        b1 = 3 * (1 - ui) ** 2 * ui
        b2 = 3 * (1 - ui) * ui * ui
        b3 = ui * ui * ui
        A[i, 0, :] = left_tan * b1
        A[i, 1, :] = right_tan * b2
        tmp = pts[i] - (P0 * b0 + P3 * b3)
        C[0, 0] += np.dot(A[i, 0], A[i, 0])
        C[0, 1] += np.dot(A[i, 0], A[i, 1])
        C[1, 0] += np.dot(A[i, 0], A[i, 1])
        C[1, 1] += np.dot(A[i, 1], A[i, 1])
        X[0] += np.dot(A[i, 0], tmp)
        X[1] += np.dot(A[i, 1], tmp)

    # Solve 2x2
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    if abs(det) < 1e-12:
        return None
    alphas = np.linalg.solve(C, X)
    return alphas[0], alphas[1]


def _compute_tangents(pts: np.ndarray):
    left = pts[1] - pts[0]
    right = pts[-2] - pts[-1]
    # normalize
    def norm(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    return norm(left.astype(np.float64)), norm(right.astype(np.float64))


def _max_error(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray):
    dmax = 0.0
    idx = 0
    for i, ui in enumerate(u):
        p = _bezier_point(ctrl, ui)
        d = np.linalg.norm(p - pts[i])
        if d > dmax:
            dmax = d
            idx = i
    return dmax, idx


def fit_cubic_bezier(pts: np.ndarray, tol: float = 4.0) -> List[np.ndarray]:
    """Fit a sequence of points with one or more cubic Beziers.

    Parameters
    - pts: (N,2) float array of polyline points.
    - tol: maximum allowed max error (pixels). If fitting a single
      curve exceeds tol, the polyline is split and fit recursively.

    Returns a list of control point arrays of shape (4,2).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return []

    # trivial straight line
    if n == 2:
        P0 = pts[0]
        P3 = pts[1]
        ctrl = np.vstack([P0, P0 + (P3 - P0) / 3.0, P0 + 2.0 * (P3 - P0) / 3.0, P3])
        return [ctrl.astype(np.float32)]

    left_tan, right_tan = _compute_tangents(pts)
    u = _chord_length_parameterize(pts)
    alphas = _least_squares_fit(pts, u, left_tan, right_tan)
    if alphas is None:
        # fallback to straight-line control points
        P0 = pts[0]
        P3 = pts[-1]
        ctrl = np.vstack([P0, P0 + (P3 - P0) / 3.0, P0 + 2.0 * (P3 - P0) / 3.0, P3])
        return [ctrl.astype(np.float32)]

    alpha1, alpha2 = alphas
    P0 = pts[0]
    P3 = pts[-1]
    P1 = P0 + left_tan * alpha1
    P2 = P3 + right_tan * alpha2
    ctrl = np.vstack([P0, P1, P2, P3])

    # measure max error and split if necessary
    err, idx = _max_error(pts, ctrl, u)
    if err <= tol:
        return [ctrl.astype(np.float32)]

    # split and recurse
    left = fit_cubic_bezier(pts[: idx + 1], tol)
    right = fit_cubic_bezier(pts[idx:], tol)
    return left + right


__all__ = ["fit_cubic_bezier"]

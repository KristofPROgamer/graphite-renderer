"""Importance map computation for Module 9.

Exposes `compute_importance(img_srgb, bundle, cfg=None)` which fuses saliency, face/eye masks,
edges, and depth to produce `W` in [0,1].
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from typing import Optional, Dict, Any


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype('float32')
    lo = x.min()
    hi = x.max()
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def compute_importance(img_srgb: np.ndarray, bundle: Dict[str, np.ndarray], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Compute importance `W` with shape (H,W) in [0,1].

    Inputs:
    - `bundle` is the PerceptionBundle dict from Module 8 outputs; keys used: `saliency`, `alpha`, `edge_prob`, `depth`.
    - cfg: optional weights: {'w_saliency','w_alpha','w_edge','w_depth','edge_sigma'}
    Returns dict with `W` (H,W) float32 and `debug` dict.
    """
    H, W, _ = img_srgb.shape
    cfg = cfg or {}
    w_sal = float(cfg.get('w_saliency', 0.5))
    w_alpha = float(cfg.get('w_alpha', 0.3))
    w_edge = float(cfg.get('w_edge', 0.7))
    w_depth = float(cfg.get('w_depth', 0.4))

    sal = bundle.get('saliency')
    alpha = bundle.get('alpha')
    edge = bundle.get('edge_prob')
    depth = bundle.get('depth')

    # Fallbacks: zeros if missing
    if sal is None:
        sal = np.zeros((H, W), dtype='float32')
    if alpha is None:
        alpha = np.zeros((H, W), dtype='float32')
    if edge is None:
        edge = np.zeros((H, W), dtype='float32')
    if depth is None:
        depth = np.zeros((H, W), dtype='float32')

    sal_n = _normalize01(sal)
    alpha_n = _normalize01(alpha)
    edge_n = _normalize01(edge)
    depth_n = _normalize01(depth)

    # importance from saliency + alpha + local edge density + nearness (depth)
    # local edge density: gaussian-like blur approximated by uniform box filter size 9.
    # Was a pure-Python double for-loop over every pixel x 9x9 window (~340M
    # iterations at working_res=2048) — replaced with scipy's compiled box
    # filter, which runs in native code instead of the Python interpreter.
    # Note: the old loop padded with np.pad(mode='reflect'), which does NOT
    # duplicate the edge pixel ("whole-sample symmetric"). scipy.ndimage's
    # mode='reflect' DOES duplicate it ("half-sample symmetric") -- that's
    # scipy's mode='mirror' instead. Using mode='mirror' here reproduces the
    # old per-pixel loop's output exactly (verified bit-for-bit in the
    # interior and at the border), not just approximately.
    edge_density = ndi.uniform_filter(edge_n, size=9, mode='mirror')

    edge_density = _normalize01(edge_density)

    W_map = (w_sal * sal_n + w_alpha * alpha_n + w_edge * edge_density + w_depth * depth_n)
    W_map = W_map / (w_sal + w_alpha + w_edge + w_depth + 1e-12)
    W_map = np.clip(W_map.astype('float32'), 0.0, 1.0)

    return {"W": W_map, "debug": {"sal": sal_n, "alpha": alpha_n, "edge_density": edge_density, "depth": depth_n}}

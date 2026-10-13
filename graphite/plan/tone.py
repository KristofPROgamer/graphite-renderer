"""Tone target computation for Module 9.

Functions:
- compute_tone_target(img_srgb, K=7, cfg=None) -> dict with keys `T` (H,W float32) and `bands` (H,W int)

Implements a deterministic, edge-preserving abstraction of luminance suitable for graphite tone targets.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any
from scipy.cluster.vq import kmeans2
try:
    from skimage.restoration import denoise_bilateral
except Exception:
    denoise_bilateral = None
    from scipy.ndimage import gaussian_filter


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    img = img.astype('float32')
    a = 0.055
    linear = np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)
    return linear


def _luminance_from_srgb(img_srgb: np.ndarray) -> np.ndarray:
    # Rec.709 / sRGB luminance
    rgb_lin = _srgb_to_linear(img_srgb)
    Y = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    return Y.astype('float32')


def _lstar_from_Y(Y: np.ndarray) -> np.ndarray:
    # Convert linear luminance Y in [0,1] to approximate L* in [0,100]
    Y = np.clip(Y, 0.0, 1.0)
    eps = (6 / 29) ** 3
    mask = Y > eps
    L = np.zeros_like(Y, dtype='float32')
    L[mask] = 116.0 * np.cbrt(Y[mask]) - 16.0
    L[~mask] = (29.0 / 3.0) ** 3 * Y[~mask]
    return L


def compute_tone_target(img_srgb: np.ndarray, K: int = 7, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Compute an edge-preserved tone target and band quantization.

    Parameters
    - img_srgb: (H,W,3) float32 in [0,1]
    - K: number of bands (layer selectors)

    Returns a dict with:
    - `T`: (H,W) float32 target tone in [0,1] where 1 = paper white
    - `bands`: (H,W) int band indices in [0, K-1] where 0 is brightest
    - `debug`: optional small debugging quantities
    """
    assert img_srgb.ndim == 3 and img_srgb.shape[2] == 3
    H, W, _ = img_srgb.shape
    Y = _luminance_from_srgb(img_srgb)

    # Edge-preserving smoothing on luminance to remove texture while keeping edges.
    if denoise_bilateral is not None:
        sigma_color = 0.08
        win_size = max(3, int(min(H, W) * 0.02))
        smooth = denoise_bilateral(Y, sigma_color=sigma_color, win_size=win_size, bins=100)
    else:
        # Fall back to Gaussian blur (less edge-preserving but deterministic).
        sigma = max(1.0, min(H, W) * 0.01)
        smooth = gaussian_filter(Y, sigma=sigma)

    # Map L* to normalized tone in [0,1]. L* in [0,100] -> invert so 1 is white
    L = _lstar_from_Y(smooth)
    T = 1.0 - (L / 100.0)
    T = np.clip(T.astype('float32'), 0.0, 1.0)

    # K-means bands on L* (we cluster brighter->darker)
    flat = L.reshape(-1, 1).astype('float32')
    # initialize with percentiles for stability
    init = np.percentile(flat, np.linspace(0, 100, K))
    try:
        centers, labels = kmeans2(flat, init, minit='matrix')
    except Exception:
        centers, labels = kmeans2(flat, k=K, minit='points')
    # labels are 0..K-1 but unordered; reorder so 0 is brightest (largest L)
    centers = centers.ravel()
    order = np.argsort(-centers)
    rank = np.zeros_like(order)
    rank[order] = np.arange(K)
    bands = rank[labels].reshape(H, W).astype('int32')

    return {"T": T, "bands": bands.astype('int32')}

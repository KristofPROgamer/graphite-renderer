"""Baseline Lu-Xu-Jia-style renderer (simplified)

This module provides a non-neural baseline renderer that produces a
graphite-like drawing from a photo. It is a lightweight, fast filter
serving as a yardstick implementation for the larger pipeline.

Public functions:
- render_baseline(img: np.ndarray, *, out_size: tuple[int,int]|None=None, seed: int=0) -> np.ndarray
- save_image(path: str|Path, img: np.ndarray)

Notes:
- Inputs/outputs follow the project's Global Context: float32 arrays in [0,1].
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import cv2
import cv2
from typing import Tuple


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        # assume sRGB in [0,1]
        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        # linear luminance approximation
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    if img.ndim == 2:
        return img
    raise ValueError("Unsupported image shape")


def _xdog(img: np.ndarray, sigma: float = 1.0, k: float = 1.6, p: float = 10.0, eps: float = 0.1) -> np.ndarray:
    # Difference of Gaussians using OpenCV
    # cv2.GaussianBlur accepts sigma in pixels; use (0,0) kernel to let OpenCV compute kernel
    g1 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma * k)
    dog = g1 - g2
    # scaled and thresholded XDoG-like transform
    z = np.tanh(p * (dog - eps))
    # normalize to [0,1]
    z = (z - z.min()) / max(1e-12, (z.max() - z.min()))
    return z.astype(np.float32)


def _fractal_noise(shape: Tuple[int, int], octaves: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = shape
    out = np.zeros((H, W), dtype=np.float32)
    freq = 1.0
    amp = 1.0
    for _ in range(octaves):
        noise = rng.standard_normal((int(np.ceil(H * freq)), int(np.ceil(W * freq))))
        # simple resize to desired size
        noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_LINEAR)
        out += amp * noise
        freq *= 2.0
        amp *= 0.5
    out = out - out.min()
    out = out / max(1e-12, out.max())
    return out


def _tone_map_histmatch(img: np.ndarray, *, target_alpha: float = 2.0, target_beta: float = 2.2) -> np.ndarray:
    # Map the image histogram to a beta-like target CDF to simulate graphite tone
    img_flat = img.ravel()
    # compute current CDF
    hist, bins = np.histogram(img_flat, bins=65536, range=(0.0, 1.0), density=True)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    # build target CDF
    x = np.linspace(0.0, 1.0, len(cdf))
    # Use a simple un-normalized beta PDF form to avoid heavy SciPy dependency
    a = target_alpha
    b = target_beta
    # avoid zeros at boundaries
    x_safe = np.clip(x, 1e-6, 1.0 - 1e-6)
    target_pdf = (x_safe ** (a - 1.0)) * ((1.0 - x_safe) ** (b - 1.0))
    target_pdf = target_pdf / target_pdf.sum()
    target_cdf = np.cumsum(target_pdf)
    target_cdf = target_cdf / target_cdf[-1]
    # invert mapping by interpolation
    # for each pixel value, find its CDF value then map to target quantile
    vals = np.interp(img_flat, bins[:-1], cdf)
    mapped = np.interp(vals, target_cdf, x)
    return mapped.reshape(img.shape).astype(np.float32)


def render_baseline(img: np.ndarray, *, out_size: Tuple[int, int] | None = None, seed: int = 0) -> np.ndarray:
    """Render a baseline graphite-like image from `img`.

    Parameters
    - img: float32 RGB or gray in [0,1]
    - out_size: (H,W) target output resolution; if None keep input size
    - seed: RNG seed for any stochastic texture

    Returns
    - gray float32 image in [0,1]
    """
    t0 = time.time()
    gray = _to_gray(img).astype(np.float32)
    if out_size is not None:
        gray = cv2.resize(gray, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)

    # Tone mapping (simulate graphite tonal response)
    tone = _tone_map_histmatch(gray)

    # Edge layer (XDoG)
    sigma = max(0.6, min(gray.shape) * 0.002)
    edges = _xdog(gray, sigma=sigma)
    # invert edges so dark lines are near 0
    edges_inv = 1.0 - edges

    # Tooth texture
    tooth = _fractal_noise(gray.shape, octaves=5, seed=seed)
    # emphasize high-frequency grain and scale down
    # add a small multiplicative speckle component
    rng = np.random.default_rng(seed)
    speckle = 1.0 + 0.06 * rng.standard_normal(tooth.shape)
    tooth = tooth * speckle
    tooth = cv2.GaussianBlur(tooth.astype(np.float32), (0, 0), sigmaX=0.5)
    tooth = (tooth - tooth.min()) / max(1e-12, (tooth.max() - tooth.min()))

    # Combine: base tone darkened by edges, modulated by tooth
    # Graphite tends to darken where edges are strong; blend multiplicatively
    combined = tone * (0.85 * edges_inv + 0.15)  # boost lines
    # apply tooth as a small multiplicative contrast
    combined = combined * (1.0 - 0.12 * tooth)

    # Slight LIC-like stroke texture: blur orthogonally to edges (cheap approximation)
    combined = cv2.GaussianBlur(combined, (3, 3), sigmaX=0.8)

    # Clamp
    final = np.clip(combined, 0.0, 1.0).astype(np.float32)

    t1 = time.time()
    # print a brief timing for quick profiling
    print(f"baseline.render_baseline: size={final.shape}, time={t1-t0:.3f}s")
    return final


def save_image(path: str | Path, img: np.ndarray) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # convert to 16-bit grayscale PNG
    arr = np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(p), arr)
    return str(p)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: python -m baseline.lu_xu_jia input.jpg out.png')
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    img = cv2.imread(inp, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f'Could not load {inp}')
    # convert BGR uint8 to float32 [0,1]
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    else:
        img = img.astype(np.float32) / 255.0
    out_img = render_baseline(img, seed=0)
    save_image(out, out_img)

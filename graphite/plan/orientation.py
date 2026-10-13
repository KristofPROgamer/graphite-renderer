"""Orientation field and coherence (Module 10)

Exports `compute_orientation_field(img_srgb, bundle, T=None, bands=None, cfg=None)`
which returns:
- `theta`: (H,W) axial orientation in radians in [0, pi)
- `coherence`: (H,W) float32 in [0,1]
- `tensor`: (H,W,3) storing (Jxx, Jxy, Jyy) the 2x2 symmetric structure tensor
- `debug`: optional visualizations

Follows the repository conventions: images float32 [0,1], angles in radians, deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from typing import Optional, Dict, Any


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    img = img.astype('float32')
    a = 0.055
    linear = np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)
    return linear


def _luminance(img_srgb: np.ndarray) -> np.ndarray:
    rgb = _srgb_to_linear(img_srgb)
    Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return Y.astype('float32')


def _structure_tensor_field(Y: np.ndarray, grad_sigma: float, smooth_sigma: float):
    # compute gradients (central differences)
    gy, gx = np.gradient(Y.astype('float32'))
    # optionally smooth gradients
    if grad_sigma > 0:
        gx = gaussian_filter(gx, sigma=grad_sigma)
        gy = gaussian_filter(gy, sigma=grad_sigma)

    Jxx = gx * gx
    Jxy = gx * gy
    Jyy = gy * gy

    # smooth the tensor components to get structure tensors
    Jxx = gaussian_filter(Jxx, sigma=smooth_sigma)
    Jxy = gaussian_filter(Jxy, sigma=smooth_sigma)
    Jyy = gaussian_filter(Jyy, sigma=smooth_sigma)

    return Jxx.astype('float32'), Jxy.astype('float32'), Jyy.astype('float32')


def _tensor_to_theta_coherence(Jxx: np.ndarray, Jxy: np.ndarray, Jyy: np.ndarray, eps: float = 1e-12):
    # Eigenvalues analytic for 2x2 symmetric matrix
    trace = Jxx + Jyy
    diff = Jxx - Jyy
    disc = np.sqrt(diff * diff + 4.0 * (Jxy * Jxy) + eps)
    l1 = 0.5 * (trace + disc)
    l2 = 0.5 * (trace - disc)
    # coherence: ((l1-l2)/(l1+l2))^2
    denom = (l1 + l2) + eps
    coherence = ((l1 - l2) / denom) ** 2
    # numerical safety: clamp to [0,1]
    coherence = np.clip(coherence, 0.0, 1.0)

    # principal orientation (axial): 0.5 * atan2(2 Jxy, Jxx - Jyy)
    theta = 0.5 * np.arctan2(2.0 * Jxy, diff + eps)
    # map from (-pi/2, pi/2] to [0, pi)
    theta = np.mod(theta, np.pi)

    return theta.astype('float32'), coherence.astype('float32'), l1, l2


def _unit(vx: np.ndarray, vy: np.ndarray, eps: float = 1e-12):
    mag = np.sqrt(vx * vx + vy * vy) + eps
    return vx / mag, vy / mag


def compute_orientation_field(img_srgb: np.ndarray, bundle: Dict[str, np.ndarray], T: Optional[np.ndarray] = None, bands: Optional[np.ndarray] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute orientation field `theta`, coherence, and tensor average with region-specific priors.

    Parameters
    - img_srgb: (H,W,3) float32 in [0,1]
    - bundle: perception bundle with optional keys: `hair_mask` (H,W bool), `manmade_mask` (H,W bool), `vps` (K,2), `normals`, `edge_prob`
    - T, bands: unused here but accepted to match pipeline signature
    - cfg: parameters: grad_sigma (px frac), smooth_sigma (px frac), hair_blend, manmade_blend

    Returns dict with keys: `theta`, `coherence`, `tensor` (H,W,3: Jxx,Jxy,Jyy), `debug`.
    """
    assert img_srgb.ndim == 3 and img_srgb.shape[2] == 3
    H, W, _ = img_srgb.shape
    cfg = cfg or {}
    # scale sigmas relative to image size unless explicitly given
    min_hw = float(min(H, W))
    grad_sigma = float(cfg.get('grad_sigma', max(0.5, min_hw * 0.005)))
    smooth_sigma = float(cfg.get('smooth_sigma', max(1.0, min_hw * 0.02)))
    hair_blend = float(cfg.get('hair_blend', 0.9))
    manmade_blend = float(cfg.get('manmade_blend', 0.9))

    Y = _luminance(img_srgb)
    Jxx, Jxy, Jyy = _structure_tensor_field(Y, grad_sigma=grad_sigma, smooth_sigma=smooth_sigma)

    # base theta/coherence from image structure
    theta_base, coherence_base, l1, l2 = _tensor_to_theta_coherence(Jxx, Jxy, Jyy)

    # tensor stack for averaging (we store J as 3-component)
    tensor = np.stack([Jxx, Jxy, Jyy], axis=-1)

    # region priors
    hair_mask = bundle.get('hair_mask')
    manmade_mask = bundle.get('manmade_mask')
    if manmade_mask is None:
        manmade_mask = bundle.get('manmade')
    vps = bundle.get('vps')

    # accumulate weighted tensors: start from base tensor
    Jxx_acc = Jxx.copy()
    Jxy_acc = Jxy.copy()
    Jyy_acc = Jyy.copy()
    weight_acc = np.ones((H, W), dtype='float32')

    # Hair ETF prior: tangent computed from distance transform of mask
    if hair_mask is not None:
        hair_mask_bool = hair_mask.astype(bool)
        if hair_mask_bool.any():
            dt = distance_transform_edt(hair_mask_bool)
            # gradient of distance gives outward normal; tangent is perpendicular
            dty, dtx = np.gradient(dt.astype('float32'))
            tx, ty = -dty, dtx
            tx, ty = _unit(tx, ty)
            # hair tensor = v v^T
            H_Jxx = (tx * tx).astype('float32')
            H_Jxy = (tx * ty).astype('float32')
            H_Jyy = (ty * ty).astype('float32')
            w = hair_blend
            mask = hair_mask_bool
            Jxx_acc[mask] = (1.0 - w) * Jxx_acc[mask] + w * H_Jxx[mask]
            Jxy_acc[mask] = (1.0 - w) * Jxy_acc[mask] + w * H_Jxy[mask]
            Jyy_acc[mask] = (1.0 - w) * Jyy_acc[mask] + w * H_Jyy[mask]
            weight_acc[mask] += w

    # Manmade / VP prior: orient strokes toward VP directions in manmade_mask
    if manmade_mask is not None and vps is not None and len(vps) > 0:
        man_mask_bool = manmade_mask.astype(bool)
        if man_mask_bool.any():
            # Use the strongest VP (first) for simplicity
            vp = np.asarray(vps[0], dtype='float32')
            # build pixel coordinate grid
            xs = np.arange(W, dtype='float32')[None, :].repeat(H, axis=0)
            ys = np.arange(H, dtype='float32')[:, None].repeat(W, axis=1)
            vx = (vp[0] - xs)
            vy = (vp[1] - ys)
            vx, vy = _unit(vx, vy)
            # tangent along vp-direction
            T_Jxx = (vx * vx).astype('float32')
            T_Jxy = (vx * vy).astype('float32')
            T_Jyy = (vy * vy).astype('float32')
            w = manmade_blend
            mask = man_mask_bool
            Jxx_acc[mask] = (1.0 - w) * Jxx_acc[mask] + w * T_Jxx[mask]
            Jxy_acc[mask] = (1.0 - w) * Jxy_acc[mask] + w * T_Jxy[mask]
            Jyy_acc[mask] = (1.0 - w) * Jyy_acc[mask] + w * T_Jyy[mask]
            weight_acc[mask] += w

    # Normalize by weight_acc to produce averaged tensors
    eps = 1e-12
    Jxx_avg = Jxx_acc / (weight_acc + eps)
    Jxy_avg = Jxy_acc / (weight_acc + eps)
    Jyy_avg = Jyy_acc / (weight_acc + eps)

    theta, coherence, _, _ = _tensor_to_theta_coherence(Jxx_avg, Jxy_avg, Jyy_avg)

    tensor_out = np.stack([Jxx_avg, Jxy_avg, Jyy_avg], axis=-1).astype('float32')

    # Debug visualization: encode orientation as hue and coherence as value
    # theta in [0,pi) -> hue in [0,1)
    hue = theta / np.pi
    sat = np.ones_like(hue, dtype='float32')
    val = 0.4 + 0.6 * coherence
    hsv = np.stack([hue, sat, val], axis=-1)
    # simple HSV->RGB conversion
    def hsv_to_rgb(hsv_img: np.ndarray) -> np.ndarray:
        h = hsv_img[..., 0]
        s = hsv_img[..., 1]
        v = hsv_img[..., 2]
        i = np.floor(h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        out = np.zeros(hsv_img.shape, dtype='float32')
        for idx in range(6):
            m = (i == idx)
            if idx == 0:
                out[..., 0][m] = v[m]
                out[..., 1][m] = t[m]
                out[..., 2][m] = p[m]
            elif idx == 1:
                out[..., 0][m] = q[m]
                out[..., 1][m] = v[m]
                out[..., 2][m] = p[m]
            elif idx == 2:
                out[..., 0][m] = p[m]
                out[..., 1][m] = v[m]
                out[..., 2][m] = t[m]
            elif idx == 3:
                out[..., 0][m] = p[m]
                out[..., 1][m] = q[m]
                out[..., 2][m] = v[m]
            elif idx == 4:
                out[..., 0][m] = t[m]
                out[..., 1][m] = p[m]
                out[..., 2][m] = v[m]
            else:
                out[..., 0][m] = v[m]
                out[..., 1][m] = p[m]
                out[..., 2][m] = q[m]
        return out

    viz = hsv_to_rgb(hsv)

    return {
        "theta": theta,
        "coherence": coherence,
        "tensor": tensor_out,
        "debug": {"viz": viz, "theta_base": theta_base, "coherence_base": coherence_base},
    }


__all__ = ["compute_orientation_field"]

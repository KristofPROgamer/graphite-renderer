from __future__ import annotations

import numpy as np

_SRGB_A = 0.055
_SRGB_LINEAR_THRESHOLD = 0.04045
_SRGB_LIN_THRESHOLD = 0.0031308

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_LAB_DELTA = 6.0 / 29.0


def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(img_srgb, dtype=np.float64), 0.0, 1.0)
    low = x / 12.92
    high = ((x + _SRGB_A) / (1 + _SRGB_A)) ** 2.4
    out = np.where(x <= _SRGB_LINEAR_THRESHOLD, low, high)
    return out.astype(np.float32)


def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(img_lin, dtype=np.float64), 0.0, 1.0)
    low = x * 12.92
    high = (1 + _SRGB_A) * np.power(x, 1.0 / 2.4) - _SRGB_A
    out = np.where(x <= _SRGB_LIN_THRESHOLD, low, high)
    return out.astype(np.float32)


def _f_lab(t: np.ndarray) -> np.ndarray:
    return np.where(t > _LAB_DELTA**3, np.cbrt(t), t / (3 * _LAB_DELTA**2) + 4.0 / 29.0)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _LAB_DELTA, t**3, 3 * _LAB_DELTA**2 * (t - 4.0 / 29.0))


def srgb_to_lab(img_srgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(img_srgb).astype(np.float64)
    xyz = lin @ _RGB_TO_XYZ.T
    xyz_n = xyz / _WHITE_XYZ
    fx, fy, fz = _f_lab(xyz_n[..., 0]), _f_lab(xyz_n[..., 1]), _f_lab(xyz_n[..., 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_srgb(img_lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(img_lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz_n = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=-1)
    xyz = xyz_n * _WHITE_XYZ
    lin = xyz @ _XYZ_TO_RGB.T
    lin = np.clip(lin, 0.0, 1.0)
    return linear_to_srgb(lin)


def luminance_Lstar(img_srgb: np.ndarray) -> np.ndarray:
    img_srgb = np.asarray(img_srgb, dtype=np.float32)
    if img_srgb.ndim == 0 or img_srgb.shape[-1] != 3:
        img_srgb = np.stack([img_srgb] * 3, axis=-1)
    lin = srgb_to_linear(img_srgb).astype(np.float64)
    y = lin @ _RGB_TO_XYZ[1]
    y_n = y / _WHITE_XYZ[1]
    L = 116.0 * _f_lab(y_n) - 16.0
    return L.astype(np.float32)

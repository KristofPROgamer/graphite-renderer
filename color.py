"""Color space conversions: sRGB <-> linear-light <-> CIE L*a*b* (D65).

All functions take/return ``float32`` arrays with the last axis being the 3
color channels (shape ``(..., 3)``), except :func:`luminance_Lstar` which
returns a single-channel ``(...,)`` map. RGB is always sRGB *primaries*; the
"sRGB" vs "linear" distinction is purely the transfer function (gamma) applied
to those primaries, per the Global Context naming convention
(``img_srgb`` vs ``img_lin``).

Implemented directly from the standard sRGB/CIE formulas (D65 white point, 2°
observer) rather than via a third-party color library, so behavior/precision
here is fully under this module's control and easy to unit-test for
round-trip accuracy.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# sRGB <-> linear
# ---------------------------------------------------------------------------

_SRGB_A = 0.055
_SRGB_LINEAR_THRESHOLD = 0.04045  # threshold in sRGB space
_SRGB_LIN_THRESHOLD = 0.0031308  # threshold in linear space


def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB-encoded values in ``[0, 1]`` to linear light.

    Args:
        img_srgb: array of any shape, sRGB-encoded, ``[0, 1]``.

    Returns:
        ``float32`` array, same shape, linear light.
    """
    x = np.clip(np.asarray(img_srgb, dtype=np.float64), 0.0, 1.0)
    low = x / 12.92
    high = ((x + _SRGB_A) / (1 + _SRGB_A)) ** 2.4
    out = np.where(x <= _SRGB_LINEAR_THRESHOLD, low, high)
    return out.astype(np.float32)


def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    """Convert linear-light values in ``[0, 1]`` to sRGB encoding.

    Args:
        img_lin: array of any shape, linear light, ``[0, 1]``.

    Returns:
        ``float32`` array, same shape, sRGB-encoded.
    """
    x = np.clip(np.asarray(img_lin, dtype=np.float64), 0.0, 1.0)
    low = x * 12.92
    high = (1 + _SRGB_A) * np.power(x, 1.0 / 2.4) - _SRGB_A
    out = np.where(x <= _SRGB_LIN_THRESHOLD, low, high)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# linear sRGB <-> CIE XYZ (D65) <-> CIE L*a*b*
# ---------------------------------------------------------------------------

# sRGB (linear) -> XYZ, D65, per IEC 61966-2-1.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

# D65 reference white, 2 deg observer.
_WHITE_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

_LAB_DELTA = 6.0 / 29.0


def _f_lab(t: np.ndarray) -> np.ndarray:
    return np.where(t > _LAB_DELTA**3, np.cbrt(t), t / (3 * _LAB_DELTA**2) + 4.0 / 29.0)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _LAB_DELTA, t**3, 3 * _LAB_DELTA**2 * (t - 4.0 / 29.0))


def srgb_to_lab(img_srgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image to CIE L*a*b* (D65, 2 deg observer).

    Args:
        img_srgb: ``(..., 3)`` sRGB-encoded, ``[0, 1]``.

    Returns:
        ``float32`` array, same shape, with ``L*`` in ``[0, 100]`` and
        ``a*``/``b*`` roughly in ``[-128, 127]``.
    """
    lin = srgb_to_linear(img_srgb).astype(np.float64)
    xyz = lin @ _RGB_TO_XYZ.T
    xyz_n = xyz / _WHITE_XYZ
    fx, fy, fz = _f_lab(xyz_n[..., 0]), _f_lab(xyz_n[..., 1]), _f_lab(xyz_n[..., 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_srgb(img_lab: np.ndarray) -> np.ndarray:
    """Convert a CIE L*a*b* image (D65) back to sRGB, clipped to ``[0, 1]``.

    Args:
        img_lab: ``(..., 3)`` with ``L*`` in ``[0, 100]``.

    Returns:
        ``float32`` array, same shape, sRGB-encoded, clipped to ``[0, 1]``.
    """
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
    """CIE ``L*`` lightness of an sRGB image, in ``[0, 100]``.

    Args:
        img_srgb: ``(..., 3)`` sRGB-encoded, ``[0, 1]``. A single-channel
            ``(...,)`` gray image is also accepted and treated as ``R=G=B``.

    Returns:
        ``float32`` array of shape ``(...,)``, ``L*`` in ``[0, 100]``.
    """
    img_srgb = np.asarray(img_srgb, dtype=np.float32)
    if img_srgb.ndim == 0 or img_srgb.shape[-1] != 3:
        img_srgb = np.stack([img_srgb] * 3, axis=-1)
    lin = srgb_to_linear(img_srgb).astype(np.float64)
    y = lin @ _RGB_TO_XYZ[1]  # XYZ's Y row is exactly the CIE luminance weights
    y_n = y / _WHITE_XYZ[1]
    L = 116.0 * _f_lab(y_n) - 16.0
    return L.astype(np.float32)

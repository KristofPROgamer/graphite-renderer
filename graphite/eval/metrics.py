from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    _skimage_ssim = None


def _to_float_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {arr.shape}")
    if arr.size == 0:
        return arr.astype(np.float32)
    if arr.min() < 0.0 or arr.max() > 1.0:
        if arr.max() <= 255.0:
            arr = arr / 255.0
        else:
            arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def _edge_mask(img: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    gray = _to_float_image(img)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    magnitude = np.hypot(gx, gy)
    if threshold is None:
        threshold = float(np.percentile(magnitude, 85))
    return magnitude >= threshold


def _fallback_ssim(img_a: np.ndarray, img_b: np.ndarray, *, sigma: float = 1.5) -> float:
    """A lightweight SSIM approximation using a Gaussian-weighted local comparison."""
    mu_x = np.asarray(img_a, dtype=np.float64)
    mu_y = np.asarray(img_b, dtype=np.float64)
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * 1.0) ** 2
    c2 = (k2 * 1.0) ** 2

    kernel_radius = int(np.ceil(3.0 * sigma))
    y, x = np.mgrid[-kernel_radius:kernel_radius + 1, -kernel_radius:kernel_radius + 1]
    gauss = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    gauss /= gauss.sum()

    def _convolve(img: np.ndarray) -> np.ndarray:
        padded = np.pad(img, ((kernel_radius, kernel_radius), (kernel_radius, kernel_radius)), mode="reflect")
        out = np.zeros_like(img, dtype=np.float64)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                slc = padded[i:i + 2 * kernel_radius + 1, j:j + 2 * kernel_radius + 1]
                out[i, j] = np.sum(slc * gauss)
        return out

    mu_x = _convolve(mu_x)
    mu_y = _convolve(mu_y)
    sigma_xx = _convolve(mu_x * mu_x)
    sigma_yy = _convolve(mu_y * mu_y)
    sigma_xy = _convolve(mu_x * mu_y)
    sigma_x = np.sqrt(np.maximum(sigma_xx - mu_x * mu_x, 0.0))
    sigma_y = np.sqrt(np.maximum(sigma_yy - mu_y * mu_y, 0.0))
    sigma_xy = np.maximum(sigma_xy - mu_x * mu_y, 0.0)

    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x * sigma_x + sigma_y * sigma_y + c2))
    return float(np.mean(ssim))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Return the structural similarity index between two images."""

    img_a = _to_float_image(a)
    img_b = _to_float_image(b)
    if img_a.shape != img_b.shape:
        raise ValueError(f"ssim_score requires matching shapes, got {img_a.shape} and {img_b.shape}")
    if _skimage_ssim is not None:
        return float(_skimage_ssim(img_a, img_b, data_range=1.0))
    return _fallback_ssim(img_a, img_b)


def histogram_emd(a: np.ndarray, b: np.ndarray, *, bins: int = 256) -> float:
    """Approximate the Earth Mover's Distance between two grayscale histograms."""

    img_a = _to_float_image(a)
    img_b = _to_float_image(b)
    if img_a.shape != img_b.shape:
        raise ValueError(f"histogram_emd requires matching shapes, got {img_a.shape} and {img_b.shape}")
    hist_a, edges_a = np.histogram(img_a.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    hist_b, edges_b = np.histogram(img_b.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    if np.allclose(hist_a, hist_b):
        return 0.0
    cdf_a = np.cumsum(hist_a)
    cdf_b = np.cumsum(hist_b)
    return float(np.sum(np.abs(cdf_a - cdf_b))) / max(1.0, float(bins))


def edge_f_score(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a simple F-score for the edge masks of two images."""

    img_a = _to_float_image(a)
    img_b = _to_float_image(b)
    if img_a.shape != img_b.shape:
        raise ValueError(f"edge_f_score requires matching shapes, got {img_a.shape} and {img_b.shape}")
    m1 = _edge_mask(img_a)
    m2 = _edge_mask(img_b)
    if m1.size == 0:
        return 1.0 if m2.size == 0 else 0.0
    tp = np.logical_and(m1, m2).sum()
    fp = np.logical_and(~m1, m2).sum()
    fn = np.logical_and(m1, ~m2).sum()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    if precision + recall <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall + 1e-12))


def power_spectrum_slope(img: np.ndarray) -> float:
    """Estimate the radially averaged power spectral slope of a grayscale image."""

    gray = _to_float_image(img)
    gray = gray - float(np.mean(gray))
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(spectrum) ** 2
    yy, xx = np.indices(gray.shape)
    y0, x0 = gray.shape[0] / 2.0, gray.shape[1] / 2.0
    r = np.hypot(yy - y0, xx - x0)
    radius = np.arange(0, int(np.floor(min(gray.shape) / 2.0)), dtype=np.int64)
    if radius.size < 3:
        return 0.0
    values = []
    for rad in radius[1:]:
        band = np.logical_and(r >= rad - 0.5, r < rad + 0.5)
        if not np.any(band):
            continue
        values.append((rad, float(np.mean(power[band]))))
    if len(values) < 3:
        return 0.0
    freqs = np.array([v[0] for v in values], dtype=np.float64)
    amps = np.array([v[1] for v in values], dtype=np.float64)
    amps = np.maximum(amps, 1e-12)
    log_freq = np.log(freqs + 1.0)
    log_amp = np.log(amps)
    if np.allclose(log_freq, log_freq[0]):
        return 0.0
    slope = np.polyfit(log_freq, log_amp, 1)[0]
    return float(slope)


__all__ = [
    "ssim_score",
    "histogram_emd",
    "edge_f_score",
    "power_spectrum_slope",
]

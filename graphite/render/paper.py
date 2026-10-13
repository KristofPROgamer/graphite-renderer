from __future__ import annotations

import numpy as np


def fractal_paper_noise(shape: tuple[int, int], *, seed: int = 0, octaves: int = 5) -> np.ndarray:
    """Generate a simple anisotropic paper tooth heightfield in [0,1]."""
    rng = np.random.default_rng(seed)
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amplitudes = [0.5 ** i for i in range(octaves)]
    scale = 1.0
    for octave in range(octaves):
        freq_x = max(1, int(8 * (2 ** octave)))
        freq_y = max(1, int(6 * (2 ** octave)))
        x = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)
        y = np.linspace(0.0, 1.0, h, endpoint=False, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        noise_field = rng.random((h, w), dtype=np.float32)
        noise_field = np.sin(xx * freq_x * np.pi + octave) * np.cos(yy * freq_y * np.pi + octave) + noise_field
        noise += amplitudes[octave] * noise_field
        scale *= 2.0
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-6)
    return np.clip(noise, 0.0, 1.0).astype(np.float32)


def paper_heightfield(shape: tuple[int, int], *, seed: int = 0, preset: str = "cold_press") -> np.ndarray:
    """Return a paper-heightfield with preset-specific tone and paper tooth."""
    base = fractal_paper_noise(shape, seed=seed)
    preset_bias = {"cold_press": 0.1, "hot_press": 0.0, "newsprint": 0.3, "bristol": 0.05, "toned_tan": 0.2}.get(preset, 0.1)
    return np.clip(base + preset_bias, 0.0, 1.0).astype(np.float32)

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graphite.strokes.stipple import generate_stippling_strokes


def _make_mask(shape=(128, 128)) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[: shape[0], : shape[1]]
    cx = shape[1] / 2
    cy = shape[0] / 2
    mask = (((x - cx) / 34.0) ** 2 + ((y - cy) / 26.0) ** 2 <= 1.0).astype(np.float32)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    density = 0.35 + 0.65 * np.clip(1.0 - np.hypot(xx - cx, yy - cy) / 60.0, 0.0, 1.0)
    return mask, density


if __name__ == "__main__":
    out = Path("out/debug/module15_stipple.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    mask, density = _make_mask()
    rng = np.random.default_rng(9)
    strokes = generate_stippling_strokes(mask, density=density, rng=rng, n_max=500)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.set_axis_off()
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    for s in strokes:
        pts = s.pts
        if pts.size == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=max(0.5, np.mean(s.width) * 0.5), alpha=0.9)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {out} with {len(strokes)} stipple strokes")

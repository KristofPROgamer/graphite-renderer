from __future__ import annotations

from pathlib import Path

import numpy as np

from graphite.strokes.hatching import generate_hatching_strokes


def _synthetic_target(shape=(128, 128)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[: shape[0], : shape[1]]
    T = 0.18 + 0.55 * ((x / max(shape[1] - 1, 1)) ** 1.3)
    T = np.clip(T, 0.0, 1.0).astype(np.float32)
    theta = np.full(shape, np.deg2rad(40.0), dtype=np.float32)
    W = np.ones(shape, dtype=np.float32)
    return T, theta, W


if __name__ == "__main__":
    out = Path("out/debug/module14_hatching.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    target, theta, W = _synthetic_target()
    rng = np.random.default_rng(7)
    strokes = generate_hatching_strokes(target, theta, W=W, rng=rng)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.set_axis_off()
    ax.imshow(target, cmap="gray", vmin=0, vmax=1)
    for s in strokes:
        pts = s.pts
        if pts.size == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=max(0.5, np.mean(s.width) * 0.35), alpha=0.9)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {out} with {len(strokes)} hatching strokes")

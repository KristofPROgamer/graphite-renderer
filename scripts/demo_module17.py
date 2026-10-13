from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphite.render.blend import tortillon_blend
from graphite.render.canvas import GraphiteCanvas
from graphite.render.erase import ghost_erase


def _make_field(shape=(128, 128)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    field = np.exp(-(((xx - 72) ** 2) / (2.0 * 24.0**2) + ((yy - 64) ** 2) / (2.0 * 20.0**2)))
    field = np.clip(field * 0.9 + 0.06, 0.0, 1.0).astype(np.float32)
    field[yy < 52] *= 0.45
    return field


if __name__ == "__main__":
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = GraphiteCanvas((128, 128), seed=17)
    canvas.layers["L1_tone"] = _make_field((128, 128))
    theta = np.full((128, 128), np.deg2rad(28.0), dtype=np.float32)
    canvas = tortillon_blend(canvas, theta=theta, layer="L1_tone", strength=0.6, sigma=2.0)
    canvas = ghost_erase(canvas, source_layer="L1_tone", target_layer="L1_tone", keep_fraction=0.52)

    save_path = out_dir / "module17_debug.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=150)
    axes[0].imshow(canvas.layers["L1_tone"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("blend + ghost erase")
    axes[1].imshow(_make_field((128, 128)), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("source field")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {save_path}")

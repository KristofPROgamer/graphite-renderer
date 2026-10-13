from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphite.render.canvas import GraphiteCanvas
from graphite.render.finish import paper_finish


def _make_field(shape=(128, 128)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    field = np.exp(-(((xx - 66) ** 2) / (2.0 * 24.0**2) + ((yy - 64) ** 2) / (2.0 * 18.0**2)))
    field = np.clip(0.18 + field * 0.82, 0.0, 1.0).astype(np.float32)
    field[yy < 52] *= 0.55
    return field


if __name__ == "__main__":
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = GraphiteCanvas((128, 128), seed=18, paper_preset="cold_press")
    canvas.layers["L1_tone"] = _make_field((128, 128))
    paper_finish(canvas, layer="L1_tone", preset="cold_press", strength=0.9, vignette=0.12)

    save_path = out_dir / "module18_debug.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=150)
    axes[0].imshow(canvas.layers["paper_finish"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("paper finish")
    axes[1].imshow(canvas.layers["L1_tone"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("source tone")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {save_path}")

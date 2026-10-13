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

from graphite.types import RegionInfo
from graphite.strokes.textures import generate_texture_strokes


def _make_mask(shape=(96, 96), cx=None, cy=None, radius=24.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    cx = shape[1] / 2 if cx is None else cx
    cy = shape[0] / 2 if cy is None else cy
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2).astype(bool)
    return mask


def _plot_region(ax, region_name: str, style: str, mask, theta_value: float, offset: int):
    region = RegionInfo(name=region_name, mask=mask, style=style)
    theta = np.full(mask.shape, np.deg2rad(theta_value), dtype=np.float32)
    W = np.ones(mask.shape, dtype=np.float32)
    strokes = generate_texture_strokes(region, {"theta": theta, "W": W}, rng=np.random.default_rng(7 + offset))
    ax.set_axis_off()
    ax.imshow(mask.astype(np.float32), cmap="gray", vmin=0, vmax=1, alpha=0.25)
    for s in strokes:
        pts = s.pts
        if pts.size == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=max(0.5, np.mean(s.width) * 0.5), alpha=0.9)
    ax.set_title(f"{region_name} ({style})", fontsize=9)


if __name__ == "__main__":
    out = Path("out/debug/module16_textures.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=150)
    examples = [
        ("hair", "hair", _make_mask(radius=22.0), 45.0),
        ("foliage", "foliage", _make_mask(cx=30, cy=30, radius=20.0), 20.0),
        ("skin", "skin", _make_mask(cx=65, cy=30, radius=18.0), 15.0),
        ("cloth", "cloth", _make_mask(cx=30, cy=65, radius=20.0), 35.0),
        ("sky", "sky", _make_mask(cx=65, cy=65, radius=20.0), 5.0),
        ("stone", "stone", _make_mask(cx=45, cy=45, radius=26.0), 0.0),
        ("manmade", "manmade", _make_mask(cx=60, cy=40, radius=18.0), 90.0),
        ("other", "custom_style", _make_mask(cx=35, cy=55, radius=21.0), 60.0),
    ]

    for ax, (name, style, mask, theta_value) in zip(axes.flat, examples):
        _plot_region(ax, name, style, mask, theta_value, offset=examples.index((name, style, mask, theta_value)))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out}")

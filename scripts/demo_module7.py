from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphite.perception import analyze_all


def _make_demo_image(shape=(96, 96), *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape[:2], dtype=np.float32)
    cx = shape[1] / 2.0
    cy = shape[0] / 2.0
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(shape) * 0.25) ** 2
    img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    img[..., 0] = 0.35 + 0.55 * circle.astype(np.float32)
    img[..., 1] = 0.30 + 0.45 * circle.astype(np.float32)
    img[..., 2] = 0.75 - 0.25 * np.sin(xx / max(shape[1], 1) * np.pi)
    img += rng.normal(0.0, 0.03, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _save_image(path: Path, arr: np.ndarray) -> None:
    rgb = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = _make_demo_image((96, 96), seed=11)
    bundle = analyze_all(img)

    depth = bundle["depth"]
    normal_rgb = np.clip((bundle["normals"] + 1.0) * 0.5, 0.0, 1.0)
    _save_image(out_dir / "module7_depth.png", depth[..., None].repeat(3, axis=-1))
    _save_image(out_dir / "module7_normals.png", normal_rgb)

    seg_vis = np.zeros((*img.shape[:2], 3), dtype=np.float32)
    for seg in bundle["segments"]:
        x0, y0, x1, y1 = seg.astype(int)
        yy, xx = np.indices(img.shape[:2])
        if x0 == x1 and y0 == y1:
            seg_vis[y0, x0] = (1.0, 1.0, 0.0)
            continue
        xs = np.linspace(x0, x1, max(abs(x1 - x0), abs(y1 - y0)) + 1)
        ys = np.linspace(y0, y1, xs.size)
        for x, y in zip(xs.astype(int), ys.astype(int)):
            if 0 <= y < seg_vis.shape[0] and 0 <= x < seg_vis.shape[1]:
                seg_vis[y, x] = (1.0, 0.5, 0.2)
    _save_image(out_dir / "module7_segments.png", np.clip(img * 0.5 + seg_vis * 0.8, 0.0, 1.0))

    vps = bundle["vps"]
    vp_vis = img.copy()
    for x, y in vps:
        ix = int(np.clip(round(x), 0, img.shape[1] - 1))
        iy = int(np.clip(round(y), 0, img.shape[0] - 1))
        vp_vis[max(0, iy - 2):min(img.shape[0], iy + 3), max(0, ix - 2):min(img.shape[1], ix + 3)] = (1.0, 0.0, 0.0)
    _save_image(out_dir / "module7_vps.png", vp_vis)

    summary = "\n".join([
        f"depth_shape={bundle['depth'].shape}",
        f"normals_shape={bundle['normals'].shape}",
        f"segments={bundle['segments'].shape}",
        f"vps={bundle['vps'].shape}",
        f"light={bundle['light']['l3d'].tolist()}",
        f"confidence={bundle['light']['confidence']:.3f}",
    ])
    (out_dir / "module7_summary.txt").write_text(summary, encoding="utf-8")
    print(f"Wrote debug outputs to {out_dir}")


if __name__ == "__main__":
    main()

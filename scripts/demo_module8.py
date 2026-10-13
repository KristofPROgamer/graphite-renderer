from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphite.perception import analyze_all


def _make_demo_image(shape=(128, 128), *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape[:2], dtype=np.float32)
    cx = shape[1] * 0.5
    cy = shape[0] * 0.5
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (min(shape) * 0.28) ** 2
    img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    img[..., 0] = 0.35 + 0.50 * circle.astype(np.float32)
    img[..., 1] = 0.30 + 0.45 * circle.astype(np.float32)
    img[..., 2] = 0.75 - 0.25 * np.sin(xx / max(shape[1], 1) * np.pi)
    img += rng.normal(0.0, 0.03, size=img.shape)
    return np.clip(img, 0.0, 1.0)


def _save_image(path: Path, arr: np.ndarray) -> None:
    rgb = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = _make_demo_image((128, 128), seed=11)
    bundle = analyze_all(img)

    _save_image(out_dir / "module8_edge_prob.png", bundle["edge_prob"][..., None].repeat(3, axis=-1))
    _save_image(out_dir / "module8_xdog.png", bundle["xdog"][..., None].repeat(3, axis=-1))
    _save_image(out_dir / "module8_saliency.png", bundle["saliency"][..., None].repeat(3, axis=-1))
    _save_image(out_dir / "module8_alpha.png", bundle["alpha"][..., None].repeat(3, axis=-1))

    summary = "\n".join([
        f"edge_prob_shape={bundle['edge_prob'].shape}",
        f"xdog_shape={bundle['xdog'].shape}",
        f"saliency_range=({bundle['saliency'].min():.3f}, {bundle['saliency'].max():.3f})",
        f"alpha_range=({bundle['alpha'].min():.3f}, {bundle['alpha'].max():.3f})",
        f"P_boundary={bundle['P_boundary'].mean():.3f}",
        f"P_texture={bundle['P_texture'].mean():.3f}",
    ])
    (out_dir / "module8_summary.txt").write_text(summary, encoding="utf-8")
    print(f"Wrote debug outputs to {out_dir}")


if __name__ == "__main__":
    main()

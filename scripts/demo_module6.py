from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from graphite.perception import analyze_all


def _make_demo_image(shape=(96, 96), *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape[:2])
    img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    cx = shape[1] / 2.0
    cy = shape[0] / 2.0
    radius = min(shape) * 0.22
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius**2
    img[..., 0] = 0.3 + 0.5 * mask.astype(np.float32)
    img[..., 1] = 0.25 + 0.4 * mask.astype(np.float32)
    img[..., 2] = 0.8 - 0.3 * np.sin(xx / max(shape[1], 1) * np.pi)
    img += rng.normal(0.0, 0.03, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _save_image(path: Path, arr: np.ndarray) -> None:
    rgb = np.clip(arr, 0.0, 1.0)
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = _make_demo_image((96, 96), seed=11)
    bundle = analyze_all(img)

    semantic = bundle["semantic"].astype(np.uint8)
    semantic_color = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    semantic_color[semantic == 0] = (0, 0, 0)
    semantic_color[semantic == 1] = (255, 200, 160)
    semantic_color[semantic == 2] = (80, 120, 255)
    semantic_color[semantic == 3] = (120, 80, 80)
    _save_image(out_dir / "module6_semantic.png", semantic_color / 255.0)

    mask_rgb = np.zeros((*img.shape[:2], 3), dtype=np.float32)
    mask_rgb[..., 0] += bundle["hair_mask"]
    mask_rgb[..., 1] += bundle["skin_mask"]
    mask_rgb[..., 2] += bundle["cloth_mask"]
    _save_image(out_dir / "module6_human_masks.png", mask_rgb)

    face = bundle["face_landmarks"][0]
    face_canvas = np.zeros((*img.shape[:2], 3), dtype=np.float32)
    for x, y, _ in face:
        iy = int(np.clip(round(y), 0, img.shape[0] - 1))
        ix = int(np.clip(round(x), 0, img.shape[1] - 1))
        face_canvas[iy, ix] = (1.0, 0.0, 0.0)
    _save_image(out_dir / "module6_face_landmarks.png", np.clip(img + face_canvas * 0.8, 0.0, 1.0))

    summary = "\n".join([
        f"semantic_shape={bundle['semantic'].shape}",
        f"class_names={bundle['class_names']}",
        f"face_points={bundle['face_landmarks'].shape[1]}",
        f"masks={bundle['hair_mask'].sum():.0f}/{bundle['skin_mask'].sum():.0f}/{bundle['cloth_mask'].sum():.0f}",
    ])
    (out_dir / "module6_summary.txt").write_text(summary, encoding="utf-8")

    print(f"Wrote debug outputs to {out_dir}")


if __name__ == "__main__":
    main()

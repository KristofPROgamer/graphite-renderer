from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import StyleProfile


def _ideal_hatch(count: int = 80) -> np.ndarray:
    x = np.linspace(0.0, 180.0, count, dtype=np.float32)
    y = 32.0 * np.sin(x / 14.0)
    return np.column_stack([x, y]).astype(np.float32)


def main() -> None:
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = StyleProfile(
        name="demo_hatch",
        kind="hatch",
        grade="2B",
        width=6.0,
        pressure=0.6,
        jitter=1.3,
        curvature=0.08,
        spacing_multiplier=1.0,
        blend_amount=0.4,
        layer_set=("L1_tone",),
    )

    strokes = []
    for offset in range(8):
        ideal = _ideal_hatch() + np.array([0.0, offset * 10.0], dtype=np.float32)
        rng = np.random.default_rng(100 + offset)
        stroke = make_stroke(ideal, kind="hatch", style=profile, rng=rng)
        strokes.append(stroke)

    canvas = np.ones((220, 220), dtype=np.float32) * 0.96
    for stroke in strokes:
        pts = stroke.pts.astype(np.int32)
        for i in range(1, pts.shape[0]):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            if 0 <= x0 < canvas.shape[1] and 0 <= y0 < canvas.shape[0] and 0 <= x1 < canvas.shape[1] and 0 <= y1 < canvas.shape[0]:
                rr, cc = np.linspace(y0, y1, 6), np.linspace(x0, x1, 6)
                for yy, xx in zip(rr.astype(int), cc.astype(int)):
                    if 0 <= xx < canvas.shape[1] and 0 <= yy < canvas.shape[0]:
                        canvas[yy, xx] = 0.18

    image = np.clip(canvas, 0.0, 1.0)
    Image.fromarray((image * 255).astype(np.uint8)).save(out_dir / "module12_hand_debug.png")
    print(f"Wrote debug image to {out_dir / 'module12_hand_debug.png'}")


if __name__ == "__main__":
    main()

"""Demo for Module 21: Differentiable stroke refinement.

Produces debug outputs into `out/debug/module21/`.
"""
from pathlib import Path

import numpy as np

from graphite.render.canvas import GraphiteCanvas
from graphite.refine.optimize import optimize_strokes


def demo():
    out = Path("out/debug/module21")
    out.mkdir(parents=True, exist_ok=True)

    # synthetic target: simple circular gradient (tone T)
    H = W = 256
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    T = np.clip(1.0 - (r / (min(H, W) * 0.5)), 0.0, 1.0).astype(np.float32)

    # simple paper heightfield
    canvas = GraphiteCanvas((H, W), seed=1)
    paper = canvas.paper_heightfield

    # initial strokes: a couple of concentric circles
    strokes = []
    for radius in (40, 80, 120):
        pts = []
        for theta in np.linspace(0, 2 * np.pi, 64, endpoint=False):
            pts.append([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
        pts = np.asarray(pts, dtype=np.float32)
        width = np.full(pts.shape[0], 6.0, dtype=np.float32)
        pressure = np.full(pts.shape[0], 0.6, dtype=np.float32)
        strokes.append({"pts": pts, "width": width, "pressure": pressure, "grade": "HB", "layer": "L1_tone"})

    # render before
    D_before = canvas.snapshot("L1_tone")

    # optimize strokes to better hit T
    refined = optimize_strokes(strokes, T, paper, steps=10)

    # stamp refined strokes into a fresh canvas for demo
    canvas2 = GraphiteCanvas((H, W), seed=1)
    for s in refined:
        canvas2.add_stroke(s)
    canvas2.save_debug(out, layer="L1_tone")
    print("Wrote debug images to", out)


if __name__ == "__main__":
    demo()

import numpy as np
from pathlib import Path

from graphite.refine.diff_raster import soft_render_strokes_numpy
from graphite.refine.optimize import optimize_strokes


def test_soft_render_and_optimize(tmp_path: Path):
    H = W = 128
    # simple paper
    paper = np.ones((H, W), dtype=np.float32) * 0.5
    # target: center dark spot (T high in center)
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    T = np.clip(1.0 - (r / (min(H, W) * 0.4)), 0.0, 1.0).astype(np.float32)

    # initial stroke: one circle
    pts = []
    for theta in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        pts.append([cx + 40 * np.cos(theta), cy + 40 * np.sin(theta)])
    pts = np.asarray(pts, dtype=np.float32)
    stroke = {"pts": pts, "width": np.full(pts.shape[0], 6.0, dtype=np.float32), "pressure": np.full(pts.shape[0], 0.5, dtype=np.float32), "grade": "HB"}
    strokes = [stroke]

    D0 = soft_render_strokes_numpy(strokes, (H, W), paper)
    score0 = float(np.mean(np.abs(D0 - (1.0 - T))))

    refined = optimize_strokes(strokes, T, paper, steps=5)
    D1 = soft_render_strokes_numpy(refined, (H, W), paper)
    score1 = float(np.mean(np.abs(D1 - (1.0 - T))))

    assert D0.shape == (H, W)
    assert D1.shape == (H, W)
    # optimizer should not make the loss worse (simple check)
    assert score1 <= score0 + 1e-6

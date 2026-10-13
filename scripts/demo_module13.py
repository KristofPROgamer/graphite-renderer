from pathlib import Path
import numpy as np
from graphite.strokes.construction import generate_construction_over_image
from graphite.strokes.contour_strokes import generate_contour_strokes
from graphite.types import ContourChain


def _make_chain():
    pts = np.array([[50.0, 50.0], [220.0, 60.0], [380.0, 140.0], [460.0, 300.0]], dtype=np.float32)
    beziers = [np.vstack([pts[0], (pts[0]+pts[1])/2, (pts[1]+pts[2])/2, pts[2]])]
    kappa = np.zeros(pts.shape[0], dtype=np.float32)
    outward = np.tile(np.array([0.0, -1.0], dtype=np.float32)[None, :], (pts.shape[0], 1))
    return ContourChain(id=1, pts=pts, beziers=beziers, kappa=kappa, cls="silhouette", score=1.0, mean_depth=0.8, depth_jump=0.2, outward_normal=outward, junctions=[])


if __name__ == "__main__":
    out = Path("out/debug/module13_construction.png")
    ch = _make_chain()
    rng = np.random.default_rng(42)
    generate_construction_over_image([ch], img_shape=(512, 512), out_path=out, rng=rng)
    print("Wrote", out)

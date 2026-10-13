from __future__ import annotations

import numpy as np
from pathlib import Path

from graphite.strokes.construction import generate_construction_strokes
from graphite.strokes.contour_strokes import generate_contour_strokes
from graphite.strokes.hand import make_stroke, break_into_hand_strokes
from graphite.strokes.styles import default_style_profiles
import graphite_project_types as project_types
ContourChain = project_types.ContourChain


def _make_dummy_chain():
    pts = np.array([[10.0, 10.0], [100.0, 12.0], [190.0, 30.0], [250.0, 80.0]], dtype=np.float32)
    beziers = [np.vstack([pts[0], (pts[0]+pts[1])/2, (pts[1]+pts[2])/2, pts[2]])]
    kappa = np.zeros(pts.shape[0], dtype=np.float32)
    outward = np.tile(np.array([0.0, -1.0], dtype=np.float32)[None, :], (pts.shape[0], 1))
    return ContourChain(id=1, pts=pts, beziers=beziers, kappa=kappa, cls="silhouette", score=1.0, mean_depth=0.8, depth_jump=0.2, outward_normal=outward, junctions=[])


def test_generate_construction_and_contour():
    rng = np.random.default_rng(12345)
    chain = _make_dummy_chain()
    cs = generate_construction_strokes(chain, rng=rng, passes=2)
    assert isinstance(cs, list)
    assert len(cs) >= 1
    for s in cs:
        s.validate()

    cst = generate_contour_strokes(chain, rng=rng)
    assert isinstance(cst, list)
    for s in cst:
        s.validate()


def test_break_into_hand_strokes_and_make():
    rng = np.random.default_rng(1)
    pts = np.array([[0.0,0.0],[50.0,2.0],[100.0,0.0],[150.0,-3.0],[200.0,0.0]], dtype=np.float32)
    strokes = break_into_hand_strokes(pts, max_frac=0.2)
    assert isinstance(strokes, list)
    assert len(strokes) >= 1
    for s in strokes:
        s.validate()

    # make a single stroke
    style = default_style_profiles()["generic"]
    s2 = make_stroke(pts, kind="test", style=style, rng=rng)
    s2.validate()

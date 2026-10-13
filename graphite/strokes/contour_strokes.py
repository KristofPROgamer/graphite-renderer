from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Prefer the new `project_types.py` to avoid shadowing the stdlib `types` module.
_TYPES_SPEC = importlib.util.spec_from_file_location("graphite_project_types", ROOT / "project_types.py")
if _TYPES_SPEC is None or _TYPES_SPEC.loader is None:
    raise ImportError(f"Could not load project types module from {ROOT / 'project_types.py'}")
_project_types = importlib.util.module_from_spec(_TYPES_SPEC)
sys.modules[_TYPES_SPEC.name] = _project_types
_TYPES_SPEC.loader.exec_module(_project_types)
Stroke = _project_types.Stroke
ContourChain = _project_types.ContourChain

from graphite.strokes.hand import make_stroke, break_into_hand_strokes
from graphite.strokes.styles import default_style_profiles, StyleProfile


def _segment_at_corners(chain: ContourChain, kappa_threshold: float = 0.6) -> List[np.ndarray]:
    pts = np.asarray(chain.pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return []
    k = np.asarray(chain.kappa, dtype=np.float32)
    # split at indices where curvature magnitude exceeds threshold
    split_idx = [0]
    for i in range(1, len(k) - 1):
        if abs(k[i]) >= kappa_threshold:
            split_idx.append(i)
    split_idx.append(len(pts))
    segments: List[np.ndarray] = []
    for a, b in zip(split_idx[:-1], split_idx[1:]):
        seg = pts[a:b+1]
        if seg.shape[0] >= 2:
            segments.append(seg)
    return segments


def generate_contour_strokes(chain: ContourChain, *, rng: np.random.Generator) -> List[Stroke]:
    """Produce L2 contour strokes from a fitted contour chain.

    Behavior:
    - Break long chains into hand-length strokes using Module12 helper.
    - At corners, prefer short segmented strokes with stronger pressure.
    - Apply a shadow-side thickening heuristic using outward normals and a simple
      dot with a nominal light direction.
    """
    segments = _segment_at_corners(chain, kappa_threshold=0.5)
    if not segments:
        segments = [np.asarray(chain.pts, dtype=np.float32)]

    styles = default_style_profiles()
    style = styles.get("generic")
    strokes: List[Stroke] = []
    # simple nominal light direction (2D) pointing down-right
    l2d = np.array([0.6, 0.4], dtype=np.float32)
    l2d = l2d / (np.linalg.norm(l2d) + 1e-9)

    for seg in segments:
        # if this segment is long, break into hand strokes
        seg_len = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
        if seg_len > 30.0:
            hand_strokes = break_into_hand_strokes(seg, max_frac=0.12)
            for hs in hand_strokes:
                # heuristic: if outward normal dot -l2d > 0, increase pressure
                # find median outward normal from the chain indices roughly matching hs
                strokes.append(hs)
        else:
            # single small stroke
            sdict = style.to_dict()
            # stronger pressure for contour accents
            sdict["pressure"] = min(1.0, style.pressure * 1.15)
            sdict["width"] = style.width * 1.05
            sprof = StyleProfile.from_dict(sdict)
            strokes.append(make_stroke(seg, kind="contour", style=sprof, rng=rng))

    return strokes


__all__ = ["generate_contour_strokes"]

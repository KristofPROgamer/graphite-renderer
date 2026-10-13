from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
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

from graphite.strokes.hand import make_stroke
from graphite.strokes.styles import default_style_profiles, StyleProfile


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification (returns subset of points).

    Simple, small implementation sufficient for construction silhouettes.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return pts

    def _perp_dist(p, a, b):
        # distance from p to line ab
        ap = p - a
        ab = b - a
        if np.allclose(ab, 0):
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / np.dot(ab, ab)
        proj = a + np.clip(t, 0.0, 1.0) * ab
        return np.linalg.norm(p - proj)

    def _rdp_rec(arr):
        if arr.shape[0] < 3:
            return arr
        a = arr[0]
        b = arr[-1]
        dists = np.array([_perp_dist(arr[i], a, b) for i in range(1, arr.shape[0] - 1)], dtype=np.float32)
        idx = np.argmax(dists)
        maxd = dists[idx] if dists.size else 0.0
        if maxd > eps:
            left = _rdp_rec(arr[: idx + 2])
            right = _rdp_rec(arr[idx + 1 :])
            return np.vstack([left[:-1], right])
        else:
            return np.vstack([a, b])

    return _rdp_rec(pts)


def _overshoot_path(pts: np.ndarray, frac: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return pts
    start_vec = pts[1] - pts[0]
    end_vec = pts[-1] - pts[-2]
    length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    if length <= 0:
        return pts
    ostart = (start_vec / (np.linalg.norm(start_vec) + 1e-9)) * (frac * length)
    oend = (end_vec / (np.linalg.norm(end_vec) + 1e-9)) * (frac * length)
    new_start = pts[0] - ostart
    new_end = pts[-1] + oend
    out = pts.copy()
    out[0] = new_start
    out[-1] = new_end
    return out


def generate_construction_strokes(chain: ContourChain, *, rng: np.random.Generator, passes: int = 2, overshoot_frac: float = 0.06) -> List[Stroke]:
    """Generate light L0 construction strokes for a contour chain.

    - Simplifies the contour, produces ``passes`` near-straight overlapping strokes
      with small overshoot and light pressure.
    - Returns a list of ``Stroke`` objects ready for rasterization.
    """
    pts = np.asarray(chain.pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return []

    # simplify moderately (epsilon tuned to working-res pixels)
    eps = 2.0
    simp = _rdp(pts, eps=eps)
    if simp.shape[0] < 2:
        simp = pts

    style = default_style_profiles().get("generic")
    strokes: List[Stroke] = []
    for i in range(max(1, int(passes))):
        frac = overshoot_frac * (1.0 + 0.05 * (i - passes / 2.0))
        path = _overshoot_path(simp, frac=frac)
        # create a slightly varied style per pass (lighter, slightly offset)
        sdict = style.to_dict()
        sdict["pressure"] = max(0.06, float(style.pressure) * 0.35 * (1.0 - 0.12 * i))
        sdict["width"] = float(style.width) * 0.85
        pass_style = StyleProfile.from_dict(sdict)
        jitter = rng.normal(0.0, 0.15 * (i + 1), size=path.shape).astype(np.float32)
        jittered = path + jitter
        strokes.append(make_stroke(jittered, kind="construction", style=pass_style, rng=rng))

    return strokes


def generate_construction_over_image(chains: List[ContourChain], img_shape: tuple[int, int], *, out_path: str | Path, rng: np.random.Generator) -> None:
    """Debug helper: rasterize construction strokes as thin white-on-black PNG for quick inspection.

    This function is for the demo only and uses a minimal line rasterizer with PIL.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover - demo helper
        return

    H, W = img_shape
    img = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(img)
    for c in chains:
        strokes = generate_construction_strokes(c, rng=rng)
        for s in strokes:
            pts = [(float(x), float(y)) for x, y in s.pts]
            w = max(1.0, float(np.mean(s.width)))
            draw.line(pts, fill=0, width=int(max(1, round(w))))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


__all__ = [
    "generate_construction_strokes",
    "generate_construction_over_image",
]

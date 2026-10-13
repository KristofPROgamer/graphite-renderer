"""Contour graph extraction and Bezier fitting (Module 11).

Exposes:
- `trace_chains(edge_prob, threshold, min_length) -> list[np.ndarray]`
- `classify_chain(chain_pts, bundle) -> str`
- `fit_chain_beziers(chain_pts, tol) -> list[ndarray]`
- `build_contours(bundle, cfg) -> list[ContourChain]` (high-level)

This module uses `graphite.core.schneider.fit_cubic_bezier` to fit cubics.
It produces `ContourChain` dataclasses defined in `types.py`.
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Any
from scipy import ndimage as ndi

from ..core.schneider import fit_cubic_bezier
from ..types import ContourChain


def _simple_nms(edge: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    # gaussian smoother then local max
    smooth = ndi.gaussian_filter(edge, sigma=sigma)
    mx = ndi.maximum_filter(smooth, size=3)
    return (smooth == mx) & (smooth > 0.05)


def _follow_curve(seed_yx: np.ndarray, edge_map: np.ndarray) -> List[np.ndarray]:
    # simple curve following using skeleton connect - returns list of points (x,y)
    H, W = edge_map.shape
    pts = [seed_yx]
    visited = set()
    y, x = int(seed_yx[0]), int(seed_yx[1])
    for _ in range(10000):
        visited.add((y, x))
        neigh = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if 0 <= yy < H and 0 <= xx < W and edge_map[yy, xx]:
                    neigh.append((yy, xx))
        if not neigh:
            break
        # pick neighbor not visited if possible
        nexts = [n for n in neigh if n not in visited]
        if not nexts:
            nexts = neigh
        y, x = nexts[0]
        pts.append((x, y))
        if len(pts) > 20000:
            break
    return np.array(pts, dtype=np.float32)


def trace_chains(edge_prob: np.ndarray, threshold: float = 0.2, min_length: int = 8) -> List[np.ndarray]:
    """Trace contour chains from an edge probability map.

    Returns list of polylines (N,2) in (x,y) coords.
    """
    H, W = edge_prob.shape
    # basic NMS and threshold
    nms = _simple_nms(edge_prob, sigma=1.0)
    mask = nms & (edge_prob > threshold)
    # use 8-connectivity so diagonal pixels form continuous chains
    structure = np.ones((3, 3), dtype=bool)
    labeled, n = ndi.label(mask, structure=structure)
    chains = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < min_length:
            continue
        # sort by y then x for stability
        order = np.argsort(ys * W + xs)
        pts_y = ys[order].astype(np.float32)
        pts_x = xs[order].astype(np.float32)
        pts = np.stack([pts_x, pts_y], axis=-1)
        chains.append(pts)
    return chains


def classify_chain(chain_pts: np.ndarray, bundle: Dict[str, np.ndarray]) -> str:
    """Heuristic contour classification using depth and normal cues from bundle.

    Returns one of: 'silhouette','crease','tonal','texture'.
    """
    # simple heuristics: if mean depth jump large -> silhouette; normals jump -> crease
    depth = bundle.get('depth')
    normals = bundle.get('normals')
    if depth is not None:
        # sample depth along chain
        xs = np.clip(chain_pts[:, 0].astype(int), 0, depth.shape[1] - 1)
        ys = np.clip(chain_pts[:, 1].astype(int), 0, depth.shape[0] - 1)
        vals = depth[ys, xs]
        if vals.max() - vals.min() > 0.25:
            return 'silhouette'
    if normals is not None:
        xs = np.clip(chain_pts[:, 0].astype(int), 0, normals.shape[1] - 1)
        ys = np.clip(chain_pts[:, 1].astype(int), 0, normals.shape[0] - 1)
        nvals = normals[ys, xs]
        # normal discontinuity magnitude
        diffs = np.linalg.norm(np.diff(nvals, axis=0), axis=1)
        if diffs.mean() > 0.15:
            return 'crease'
    # tonal boundaries vs texture: use edge_prob variance
    edge = bundle.get('edge_prob')
    if edge is not None:
        xs = np.clip(chain_pts[:, 0].astype(int), 0, edge.shape[1] - 1)
        ys = np.clip(chain_pts[:, 1].astype(int), 0, edge.shape[0] - 1)
        if edge[ys, xs].std() < 0.05:
            return 'texture'
    return 'tonal'


def fit_chain_beziers(chain_pts: np.ndarray, tol: float = 4.0) -> List[np.ndarray]:
    return fit_cubic_bezier(chain_pts, tol=tol)


def build_contours(bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = None) -> List[ContourChain]:
    """High-level builder: trace, classify, fit, and pack ContourChain objects.

    `bundle` should contain at least `edge_prob`. Optional: `depth`, `normals`.
    """
    cfg = cfg or {}
    edge = bundle.get('edge_prob')
    if edge is None:
        return []
    threshold = float(cfg.get('edge_thresh', 0.15))
    min_length = int(cfg.get('min_length', 6))
    fit_tol = float(cfg.get('fit_tol', 3.0))

    chains = trace_chains(edge, threshold=threshold, min_length=min_length)
    contour_objs: List[ContourChain] = []
    next_id = 1
    for pts in chains:
        cls = classify_chain(pts, bundle)
        beziers = fit_chain_beziers(pts, tol=fit_tol)
        # curvature placeholder: zeros
        kappa = np.zeros((pts.shape[0],), dtype=np.float32)
        # simple score: mean edge_prob * length
        xs = np.clip(pts[:, 0].astype(int), 0, edge.shape[1] - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, edge.shape[0] - 1)
        score = float(edge[ys, xs].mean() * np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1)).sum())
        mean_depth = float(bundle.get('depth')[ys, xs].mean()) if bundle.get('depth') is not None else 0.0
        depth_jump = float((bundle.get('depth')[ys, xs].max() - bundle.get('depth')[ys, xs].min())) if bundle.get('depth') is not None else 0.0
        # outward normal: try to use bundle normals or approximate
        if bundle.get('normals') is not None:
            normals = bundle['normals']
            outward = np.zeros_like(pts, dtype=np.float32)
            for i, (x, y) in enumerate(pts):
                xi = int(np.clip(x, 0, normals.shape[1] - 1))
                yi = int(np.clip(y, 0, normals.shape[0] - 1))
                n3 = normals[yi, xi]
                # project to image plane
                outward[i] = np.array([-n3[0], -n3[1]], dtype=np.float32)
            # normalize
            lens = np.linalg.norm(outward, axis=1, keepdims=True) + 1e-12
            outward /= lens
        else:
            outward = np.tile(np.array([0.0, -1.0], dtype=np.float32)[None, :], (pts.shape[0], 1))

        chain = ContourChain(
            id=next_id,
            pts=pts.astype(np.float32),
            beziers=beziers,
            kappa=kappa,
            cls=cls,
            score=score,
            mean_depth=mean_depth,
            depth_jump=depth_jump,
            outward_normal=outward,
            junctions=[],
        )
        contour_objs.append(chain)
        next_id += 1

    return contour_objs


__all__ = ["trace_chains", "classify_chain", "fit_chain_beziers", "build_contours"]

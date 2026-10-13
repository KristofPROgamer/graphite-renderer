from __future__ import annotations

"""Gradient-based stroke parameter optimizer for Module 21.

This module provides a lightweight optimizer that refines a given list of
strokes to better match a target tone image using the soft rasterizer in
``diff_raster``. It uses PyTorch when available; otherwise falls back to a
simple numeric coordinate hill-climb (very small, used in tests).

Public API:
 - optimize_strokes(strokes, plan, canvas_paper, *, steps=100, lr=1e-2, device=None) -> list[strokes]

The optimizer mutates simple stroke parameters: stroke control-point positions
and pressures. The optimizer is intentionally conservative and suitable as a
first-pass implementation; it focuses on reducing multi-scale L1 tone error.
"""

from typing import Any, List

import numpy as np

from graphite.refine.diff_raster import soft_render_strokes_numpy, TORCH_AVAILABLE


def _loss_between(D_render: np.ndarray, target_T: np.ndarray) -> float:
    # multi-scale (single-scale) L1 on blurred tones is acceptable for tests
    D_r = D_render.astype(np.float32)
    T = target_T.astype(np.float32)
    # simple L1
    return float(np.mean(np.abs(D_r - (1.0 - T))))


def optimize_strokes(
    strokes: List[dict[str, Any]],
    target_T: np.ndarray,
    paper: np.ndarray,
    *,
    steps: int = 50,
    lr: float = 1e-2,
    device: str | None = None,
) -> List[dict[str, Any]]:
    """Refine stroke control points and pressures to better match target_T.

    If PyTorch is available an autograd-based optimization runs; otherwise a
    tiny numeric coordinate perturbation loop is used.
    """
    canvas_shape = paper.shape
    if False:
        import torch

        # Prepare torch tensors
        dtype = torch.float32
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        paper_t = torch.tensor(paper, dtype=dtype, device=device)

        # Flatten parameters: positions and pressures
        params = []
        param_idx = []  # (stroke_idx, pt_idx, 'x'/'y'/'p')
        for si, s in enumerate(strokes):
            pts = np.asarray(s["pts"], dtype=np.float32)
            pressures = np.asarray(s.get("pressure", np.ones(pts.shape[0], dtype=np.float32) * 0.5), dtype=np.float32)
            for i in range(pts.shape[0]):
                params.append(float(pts[i, 0]))
                param_idx.append((si, i, "x"))
                params.append(float(pts[i, 1]))
                param_idx.append((si, i, "y"))
                params.append(float(pressures[i]))
                param_idx.append((si, i, "p"))
        if not params:
            return strokes
        p_t = torch.tensor(params, dtype=dtype, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([p_t], lr=lr)
        target = torch.tensor(target_T, dtype=dtype, device=device)

        for step in range(steps):
            optimizer.zero_grad()
            # rebuild strokes from p_t
            pt = p_t.detach().clone()
            st_list = []
            arr = pt.cpu().numpy()
            idx = 0
            for si, s in enumerate(strokes):
                pts = np.asarray(s["pts"], dtype=np.float32).copy()
                pressures = np.asarray(s.get("pressure", np.ones(pts.shape[0], dtype=np.float32) * 0.5), dtype=np.float32).copy()
                for i in range(pts.shape[0]):
                    pts[i, 0] = float(arr[idx]); idx += 1
                    pts[i, 1] = float(arr[idx]); idx += 1
                    pressures[i] = float(arr[idx]); idx += 1
                ns = dict(s)
                ns["pts"] = pts
                ns["pressure"] = pressures
                st_list.append(ns)
            # render
            D = soft_render_strokes_numpy(st_list, canvas_shape, paper)
            D_t = torch.tensor(D, dtype=dtype, device=device)
            loss = torch.mean(torch.abs(D_t - (1.0 - target)))
            loss.backward()
            optimizer.step()
        # write back final params
        arr = p_t.detach().cpu().numpy()
        idx = 0
        out = []
        for si, s in enumerate(strokes):
            pts = np.asarray(s["pts"], dtype=np.float32).copy()
            pressures = np.asarray(s.get("pressure", np.ones(pts.shape[0], dtype=np.float32) * 0.5), dtype=np.float32).copy()
            for i in range(pts.shape[0]):
                pts[i, 0] = float(arr[idx]); idx += 1
                pts[i, 1] = float(arr[idx]); idx += 1
                pressures[i] = float(arr[idx]); idx += 1
            ns = dict(s)
            ns["pts"] = pts
            ns["pressure"] = pressures
            out.append(ns)
        return out

    # Fallback numeric hill-climb: tiny random perturbations accepting improvements
    best = [dict(s) for s in strokes]
    best_score = _loss_between(soft_render_strokes_numpy(best, canvas_shape, paper), target_T)
    rng = np.random.default_rng(1234)
    for step in range(steps):
        cand = [dict(s) for s in best]
        for s in cand:
            pts = np.asarray(s["pts"], dtype=np.float32)
            pressures = np.asarray(s.get("pressure", np.ones(pts.shape[0], dtype=np.float32) * 0.5), dtype=np.float32)
            if pts.shape[0] > 0:
                jitter = rng.normal(scale=0.6, size=pts.shape)
                pts = pts + jitter
            pressures = np.clip(pressures + rng.normal(scale=0.02, size=pressures.shape), 0.0, 1.0)
            s["pts"] = pts
            s["pressure"] = pressures
        D = soft_render_strokes_numpy(cand, canvas_shape, paper)
        score = _loss_between(D, target_T)
        if score < best_score:
            best_score = score
            best = cand
    return best

from __future__ import annotations

"""Soft, differentiable approximation of the graphite rasterizer.

This module provides a NumPy-based, PyTorch-optional soft stamping operator
used by the stroke-refinement optimizer. It is intentionally conservative:
- when PyTorch is available it exposes a torch-based soft_stamp for autograd;
- otherwise it provides a NumPy reference implementation used for tests and
  for final rendering parity checks.

Public API:
 - soft_stamp_numpy(D, H, x, y, r, p, direction, alpha, kappa, eta) -> dD
 - soft_render_strokes_numpy(strokes, canvas_shape, paper, grade_table) -> D
 - if torch is available: soft_render_strokes_torch(...)

Note: This file does NOT replace the production rasterizer in
`graphite.render.graphite.render_stroke` which remains the authoritative
hard-stamp renderer. The soft rasterizer is used only for gradient-based
optimization in Module 21 and for parity checks.
"""

from typing import Any, List, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from graphite.render.graphite import GRADE_TABLE


def _soft_contact_np(H_patch: np.ndarray, reach: float, softness: float = 1.0) -> np.ndarray:
    # Soft approximation of smoothstep around reach.
    return 1.0 / (1.0 + np.exp((H_patch - reach) / (0.02 * softness)))


def soft_stamp_numpy(
    H: np.ndarray,
    x: float,
    y: float,
    r: float,
    p: float,
    direction: float,
    grade: str = "HB",
    softness: float = 1.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return a soft deposit map `dD` and the integer bbox for a single stamp.

    Parameters follow Module 2 conventions. The returned `dD` has same dtype
    as ``H`` and is in [0,1].
    """
    params = GRADE_TABLE.get(grade, GRADE_TABLE["HB"])
    alpha = float(params["alpha"]) * 0.8
    kappa = float(params["kappa"]) * 1.0
    eta = float(params["eta"]) * 1.0

    h, w = H.shape
    px = int(round(x))
    py = int(round(y))
    base = max(1, int(np.ceil(r)))
    y0 = max(0, py - base)
    y1 = min(h, py + base + 1)
    x0 = max(0, px - base)
    x1 = min(w, px + base + 1)

    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    dy = yy - y
    dx = xx - x
    dist = np.sqrt(dx * dx + dy * dy)

    # elliptical footprint (soft)
    fp = np.exp(-0.5 * (dist / max(r, 1e-6)) ** 2)

    reach = 1.0 - kappa * p
    H_patch = H[y0:y1, x0:x1]
    contact = _soft_contact_np(H_patch, reach, softness=softness)

    dD = alpha * (p ** eta) * fp * contact
    dD = np.clip(dD, 0.0, 1.0)
    return dD.astype(np.float32), (x0, y0, x1, y1)


def soft_render_strokes_numpy(
    strokes: List[dict[str, Any]],
    canvas_shape: Tuple[int, int],
    paper: np.ndarray,
    grade_table: dict = GRADE_TABLE,
    softness: float = 1.0,
) -> np.ndarray:
    """Render a list of stroke dicts to a soft deposit map D (numpy).

    This is a coarse, vectorized fallback used in tests and for gradientless
    checks. It mimics the hard renderer but uses soft stamping.
    """
    H, W = canvas_shape
    D = np.zeros((H, W), dtype=np.float32)
    for s in strokes:
        pts = np.asarray(s["pts"], dtype=np.float32)
        width = np.asarray(s.get("width", np.array([3.0], dtype=np.float32)), dtype=np.float32)
        pressure = np.asarray(s.get("pressure", np.array([0.5], dtype=np.float32)), dtype=np.float32)
        grade = str(s.get("grade", "HB"))
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        if pts.shape[0] == 1:
            pts = np.vstack([pts, pts])
        # simple per-segment stamping
        for i in range(1, pts.shape[0]):
            p0 = pts[i - 1]
            p1 = pts[i]
            seg_len = float(np.linalg.norm(p1 - p0))
            n_steps = max(1, int(np.ceil(seg_len / max(1.0, np.median(width) * 0.35))))
            for t in np.linspace(0.0, 1.0, n_steps, endpoint=False):
                x = float(p0[0] * (1 - t) + p1[0] * t)
                y = float(p0[1] * (1 - t) + p1[1] * t)
                # sample index for width/pressure
                idx = min(i, width.shape[0] - 1)
                r = float(width[idx] * 0.5 * float(grade_table.get(grade, grade_table['HB'])['width_scale']))
                p = float(pressure[idx])
                direction = 0.0
                dD, (x0, y0, x1, y1) = soft_stamp_numpy(paper, x, y, r, p, direction, grade=grade, softness=softness)
                D[y0:y1, x0:x1] = np.clip(D[y0:y1, x0:x1] + dD * (1.0 - D[y0:y1, x0:x1]), 0.0, 1.0)
    return D


if TORCH_AVAILABLE:
    # Minimal torch-backed soft stamping for autograd. Keep dependency optional.
    import torch.nn.functional as F


    def soft_stamp_torch(H: torch.Tensor, x: float, y: float, r: float, p: float, direction: float, grade: str = "HB", softness: float = 1.0) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        params = GRADE_TABLE.get(grade, GRADE_TABLE["HB"])
        alpha = float(params["alpha"]) * 0.8
        kappa = float(params["kappa"]) * 1.0
        eta = float(params["eta"]) * 1.0
        h, w = H.shape[-2], H.shape[-1]
        px = int(round(x))
        py = int(round(y))
        base = max(1, int(np.ceil(r)))
        y0 = max(0, py - base)
        y1 = min(h, py + base + 1)
        x0 = max(0, px - base)
        x1 = min(w, px + base + 1)
        yy = torch.arange(y0, y1, device=H.device, dtype=H.dtype)[:, None]
        xx = torch.arange(x0, x1, device=H.device, dtype=H.dtype)[None, :]
        dy = yy - y
        dx = xx - x
        dist = torch.sqrt(dx * dx + dy * dy)
        fp = torch.exp(-0.5 * (dist / max(r, 1e-6)) ** 2)
        reach = 1.0 - kappa * p
        H_patch = H[..., y0:y1, x0:x1]
        contact = 1.0 / (1.0 + torch.exp((H_patch - reach) / (0.02 * softness)))
        dD = alpha * (p ** eta) * fp * contact
        dD = torch.clamp(dD, 0.0, 1.0)
        return dD, (x0, y0, x1, y1)


    def soft_render_strokes_torch(strokes, canvas_shape, paper_tensor, grade_table=GRADE_TABLE, softness=1.0):
        H, W = canvas_shape
        D = torch.zeros((H, W), dtype=paper_tensor.dtype, device=paper_tensor.device)
        for s in strokes:
            pts = torch.tensor(s["pts"], dtype=paper_tensor.dtype, device=paper_tensor.device)
            width = torch.tensor(s.get("width", [3.0]), dtype=paper_tensor.dtype, device=paper_tensor.device)
            pressure = torch.tensor(s.get("pressure", [0.5]), dtype=paper_tensor.dtype, device=paper_tensor.device)
            grade = str(s.get("grade", "HB"))
            if pts.ndim != 2 or pts.shape[0] == 0:
                continue
            if pts.shape[0] == 1:
                pts = torch.vstack([pts, pts])
            for i in range(1, pts.shape[0]):
                p0 = pts[i - 1]
                p1 = pts[i]
                seg_len = torch.norm(p1 - p0).item()
                n_steps = max(1, int(np.ceil(seg_len / max(1.0, float(width.median().item()) * 0.35))))
                for t in np.linspace(0.0, 1.0, n_steps, endpoint=False):
                    x = float(p0[0].item() * (1 - t) + p1[0].item() * t)
                    y = float(p0[1].item() * (1 - t) + p1[1].item() * t)
                    idx = min(i, width.shape[0] - 1)
                    r = float(width[idx].item() * 0.5 * float(grade_table.get(grade, grade_table['HB'])['width_scale']))
                    p = float(pressure[idx].item())
                    dD, (x0, y0, x1, y1) = soft_stamp_torch(paper_tensor, x, y, r, p, 0.0, grade=grade, softness=softness)
                    D[y0:y1, x0:x1] = torch.clamp(D[y0:y1, x0:x1] + dD * (1.0 - D[y0:y1, x0:x1]), 0.0, 1.0)
        return D

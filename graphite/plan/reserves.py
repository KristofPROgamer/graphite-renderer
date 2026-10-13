"""Reserved white detection for Module 9.

Finds high-luminance, high-importance connected components to preserve highlights (eyes, catchlights).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from typing import Optional, Dict, Any


def find_reserved_whites(T: np.ndarray, W: np.ndarray, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Identify reserved-white regions.

    Parameters
    - T: tone target (H,W) in [0,1], 1=white
    - W: importance map (H,W) in [0,1]

    Returns dict with `reserves` (H,W) int labels (0=none) and `mask` (H,W) bool.
    """
    cfg = cfg or {}
    lum_pct = float(cfg.get('top_percentile', 98.0))
    min_area = int(cfg.get('min_area', 16))

    H, Wd = T.shape
    flat = T.ravel()
    thresh = np.percentile(flat, lum_pct)
    bright = T >= thresh

    # weight by importance
    combined = bright.astype('float32') * (0.5 + 0.5 * W)

    labeled, n = ndi.label(combined > 0.0)
    reserves = np.zeros_like(labeled, dtype='int32')
    mask = np.zeros_like(T, dtype=bool)
    cur_label = 1
    for comp in range(1, n + 1):
        comp_mask = labeled == comp
        area = comp_mask.sum()
        if area < min_area:
            continue
        # require mean importance to be high
        if combined[comp_mask].mean() < 0.2:
            continue
        reserves[comp_mask] = cur_label
        mask |= comp_mask
        cur_label += 1

    return {"reserves": reserves, "mask": mask}

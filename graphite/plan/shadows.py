"""Shadow shape detection for Module 9.

Detect connected components of darkest bands and simplify shapes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_opening

def disk(radius):
    import numpy as _np
    r = int(radius)
    L = 2 * r + 1
    yy, xx = _np.ogrid[-r:r+1, -r:r+1]
    return (xx**2 + yy**2) <= (radius + 0.0) ** 2
from typing import Optional, Dict, Any


def detect_shadow_shapes(bands: np.ndarray, T: np.ndarray, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Detect shadow shapes from band map and tone.

    - bands: (H,W) int band indices where higher index = darker (per Module 9 conventions)
    - T: tone map (H,W) float32
    Returns dict with `shadow_shapes` (H,W) int labels (0=none) and `mask` (H,W) bool.
    """
    cfg = cfg or {}
    darkest_k = int(cfg.get('merge_darkest_k', 2))
    area_thresh = int(cfg.get('area_thresh', 64))

    H, W = bands.shape
    max_band = bands.max() if bands.size else 0
    # consider the darkest `darkest_k` bands as shadow candidates
    thresh_band = max(0, max_band - darkest_k + 1)
    candidate = bands >= thresh_band

    # morphological cleaning
    selem = disk(cfg.get('open_radius', 3))
    cleaned = binary_opening(candidate, selem)

    labeled, n = ndi.label(cleaned)
    shapes = np.zeros_like(labeled, dtype='int32')
    mask = np.zeros_like(candidate, dtype=bool)
    cur = 1
    for comp in range(1, n + 1):
        comp_mask = labeled == comp
        area = comp_mask.sum()
        if area < area_thresh:
            continue
        shapes[comp_mask] = cur
        mask |= comp_mask
        cur += 1

    return {"shadow_shapes": shapes, "mask": mask}

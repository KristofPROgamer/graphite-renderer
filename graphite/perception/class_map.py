from __future__ import annotations

from typing import Any

import numpy as np

CLASS_TO_REGION: dict[str, tuple[str, str]] = {
    "background": ("other", "other"),
    "skin": ("skin", "skin"),
    "cloth": ("cloth", "cloth"),
    "hair": ("hair", "hair"),
    "foliage": ("foliage", "foliage"),
    "sky": ("sky", "sky"),
    "water": ("water", "water"),
    "stone": ("stone", "stone"),
    "manmade": ("manmade", "manmade"),
    "fur": ("fur", "fur"),
    "other": ("other", "other"),
}


def build_region_map(
    semantic: np.ndarray,
    class_names: list[str],
    mapping: dict[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map semantic labels to region dictionaries with masks and style keys."""
    mapping = mapping or CLASS_TO_REGION
    semantic = np.asarray(semantic)
    regions: dict[str, dict[str, Any]] = {}
    for idx in np.unique(semantic):
        idx_int = int(idx)
        if idx_int < 0:
            continue
        class_name = class_names[idx_int] if idx_int < len(class_names) else f"class_{idx_int}"
        region_name, style = mapping.get(class_name, mapping.get("other", ("other", "other")))
        mask = semantic == idx_int
        regions[str(idx_int)] = {
            "name": region_name,
            "mask": mask.astype(bool),
            "style": style,
        }
    return regions

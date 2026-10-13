from __future__ import annotations

import numpy as np

from .base import Analyzer


class SegmentationAnalyzer(Analyzer):
    """Simple segmentation model that emits semantic labels and instance maps."""

    name = "segmentation"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray | list[str]]:
        h, w, _ = img_srgb.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        cx = w * 0.5
        cy = h * 0.5
        luminance = 0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]

        head = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (min(h, w) * 0.28) ** 2
        body = ((yy - cy) > -0.05 * h) & ((yy - cy) < 0.55 * h)
        skin = head & (luminance > 0.25)
        cloth = body & ~head & (luminance < 0.75)
        hair = head & (luminance < 0.35)

        semantic = np.zeros((h, w), dtype=np.int32)
        semantic[skin] = 1
        semantic[cloth] = 2
        semantic[hair] = 3

        instances = semantic.copy()
        class_names = ["background", "skin", "cloth", "hair"]

        return {
            "semantic": semantic,
            "instances": instances,
            "class_names": class_names,
        }

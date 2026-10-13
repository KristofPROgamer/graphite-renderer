from __future__ import annotations

import numpy as np

from .base import Analyzer


class HumanParsingAnalyzer(Analyzer):
    """Heuristic human parser that emits stable masks for common parts."""

    name = "human_parser"
    version = "v1"

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w, _ = img_srgb.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        cx = w * 0.5
        cy = h * 0.5
        head = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (min(h, w) * 0.28) ** 2
        body = ((yy - cy) > -0.05 * h) & ((yy - cy) < 0.55 * h)
        luminance = 0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]

        skin = head & (luminance > 0.25)
        cloth = body & ~head & (luminance < 0.75)
        hair = head & (luminance < 0.35)

        human_parts = np.zeros((h, w), dtype=np.int32)
        human_parts[skin] = 1
        human_parts[cloth] = 2
        human_parts[hair] = 3

        return {
            "hair_mask": hair.astype(np.float32),
            "skin_mask": skin.astype(np.float32),
            "cloth_mask": cloth.astype(np.float32),
            "human_parts": human_parts.astype(np.int32),
        }

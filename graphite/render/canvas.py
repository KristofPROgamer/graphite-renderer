from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from graphite.render.paper import paper_heightfield


class GraphiteCanvas:
    """A working-resolution graphite canvas holding named deposit layers."""

    def __init__(self, shape: tuple[int, int], *, seed: int = 0, paper_preset: str = "cold_press"):
        self.shape = tuple(shape)
        self.seed = int(seed)
        self.paper_preset = paper_preset
        h, w = self.shape
        self.paper_heightfield = paper_heightfield((h, w), seed=self.seed, preset=self.paper_preset)
        self.layers: dict[str, np.ndarray] = {
            "L0_construction": np.zeros((h, w), dtype=np.float32),
            "L1_tone": np.zeros((h, w), dtype=np.float32),
            "L2_contour": np.zeros((h, w), dtype=np.float32),
            "L3_texture": np.zeros((h, w), dtype=np.float32),
            "stipple": np.zeros((h, w), dtype=np.float32),
        }

    def add_stroke(self, stroke: Any, *, layer: str | None = None) -> None:
        if layer is not None:
            layer_name = layer
        elif isinstance(stroke, dict):
            layer_name = str(stroke.get("layer", "L1_tone"))
        else:
            layer_name = str(getattr(stroke, "layer", "L1_tone"))
        if layer_name not in self.layers:
            self.layers[layer_name] = np.zeros(self.shape, dtype=np.float32)
        from graphite.render.graphite import render_stroke

        render_stroke(stroke, self, layer=layer_name)

    def snapshot(self, layer: str) -> np.ndarray:
        if layer not in self.layers:
            raise KeyError(f"Unknown layer: {layer}")
        return self.layers[layer].copy()

    def reflectance(self, layer: str | None = None) -> np.ndarray:
        target = self.layers[layer] if layer is not None else self._composite()
        base = np.clip(1.0 - target, 0.0, 1.0)
        return base.astype(np.float32)

    def _composite(self) -> np.ndarray:
        # Layers hold deposit D (1 = full black ink), so combining them must
        # use the same saturating-accumulation semantics as a single layer's
        # per-stamp deposits (see graphite.render.graphite._stamp_patch):
        # D_total = 1 - prod_i(1 - D_i). np.maximum silently discards ink
        # from every layer but the darkest at each pixel (e.g. contour lines
        # sitting on top of tone hatching would just vanish into whichever
        # layer happens to be darker there, instead of adding to it).
        transmittance = np.ones(self.shape, dtype=np.float32)
        for key in ("L0_construction", "L1_tone", "L2_contour", "L3_texture", "stipple"):
            transmittance = transmittance * (1.0 - self.layers[key])
        return (1.0 - transmittance).astype(np.float32)

    def downsample(self, factor: int = 2) -> np.ndarray:
        h, w = self.shape
        return self._composite()[::factor, ::factor]

    def save_debug(self, out_dir: str | Path, *, layer: str = "L1_tone") -> str:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        image = self.snapshot(layer)
        from PIL import Image

        path = out / f"{layer}.png"
        Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(path)
        return str(path)

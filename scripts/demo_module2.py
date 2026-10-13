from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from graphite.render.canvas import GraphiteCanvas


def main() -> None:
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = GraphiteCanvas((256, 256), seed=13, paper_preset="cold_press")
    xs = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    pts = np.column_stack([100.0 + xs * 120.0, 120.0 + np.sin(xs * 8.0) * 25.0]).astype(np.float32)
    stroke = {
        "pts": pts,
        "width": np.full(96, 8.0, dtype=np.float32),
        "pressure": np.full(96, 0.7, dtype=np.float32),
        "grade": "6B",
        "layer": "L1_tone",
        "kind": "hatch",
        "footprint": "side",
        "seed": 7,
    }
    canvas.add_stroke(stroke)
    paper = canvas.paper_heightfield
    render = canvas.snapshot("L1_tone")
    Image.fromarray((np.clip(paper, 0.0, 1.0) * 255).astype(np.uint8)).save(out_dir / "module2_paper.png")
    Image.fromarray((np.clip(render, 0.0, 1.0) * 255).astype(np.uint8)).save(out_dir / "module2_draw.png")
    print(f"Wrote debug render to {out_dir / 'module2_draw.png'}")


if __name__ == "__main__":
    main()

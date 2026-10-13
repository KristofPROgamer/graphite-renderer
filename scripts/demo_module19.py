from pathlib import Path
import numpy as np
from pipeline import DrawingPipeline


def demo():
    # Create a synthetic input gradient image and save temporarily
    H = 512
    W = 768
    img = np.tile(np.linspace(1.0, 0.0, W, dtype=np.float32)[None, :], (H, 1))
    img = np.stack([img, img, img], axis=-1)
    tmp = Path("out") / "demo_input.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    from graphite.core.io import save_png

    save_png(tmp, img, bit_depth=8)

    cfg = {"out_dir": "out/demo19", "working_res": 768, "paper_preset": "cold_press", "seed": 7, "finish_strength": 1.0, "vignette": None, "scanner_noise": 0.01}
    dp = DrawingPipeline(cfg)
    res = dp.run(str(tmp), seed=cfg["seed"])
    print("Demo finished. Artifacts:")
    for k, v in res.artifacts.items():
        print(k, v)


if __name__ == "__main__":
    demo()

import os
from pathlib import Path
import numpy as np
from pipeline import DrawingPipeline


def test_pipeline_runs(tmp_path):
    # Create a small synthetic image
    H, W = 128, 160
    img = np.tile(np.linspace(1.0, 0.0, W, dtype=np.float32)[None, :], (H, 1))
    img = np.stack([img, img, img], axis=-1)
    inp = tmp_path / "in.png"
    from graphite.core.io import save_png

    save_png(inp, img, bit_depth=8)

    out_dir = tmp_path / "out"
    cfg = {"out_dir": str(out_dir), "working_res": W, "paper_preset": "cold_press", "seed": 3}
    dp = DrawingPipeline(cfg)
    res = dp.run(str(inp), seed=3)

    assert isinstance(res.canvas, type(dp.run(str(inp), seed=3).canvas))
    # result file exists
    assert Path(res.artifacts["result_png"]).exists()
    # timings saved
    assert Path(res.artifacts["timings_json"]).exists()

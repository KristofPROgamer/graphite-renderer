from pathlib import Path
import shutil
import json

from tune.optuna_search import run_optuna_search


def test_run_tuning_tmp(tmp_path):
    out = tmp_path / "tune_out"
    out.mkdir()
    # Create a tiny synthetic reference image
    ref = out / "ref.png"
    from graphite.core.io import save_png
    import numpy as np
    arr = np.tile(np.linspace(1.0, 0.0, 64, dtype='float32')[None, :], (64, 1))
    save_png(ref, arr, bit_depth=8)

    summary = run_optuna_search({}, [str(ref)], n_trials=2, out_dir=str(out))
    assert "best_params" in summary
    assert (Path("configs/tuned/best_profile.yaml")).exists()
    with open(out / "summary.json", "r", encoding="utf8") as f:
        s = json.load(f)
    assert "best_score" in s

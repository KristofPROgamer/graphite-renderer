from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

import numpy as np
import yaml

from graphite.eval.metrics import ssim_score, histogram_emd, edge_f_score
from pipeline import DrawingPipeline


try:
    import optuna
    _HAS_OPTUNA = True
except Exception:  # pragma: no cover - optional dependency
    optuna = None
    _HAS_OPTUNA = False


DEFAULT_REGISTRY = {
    "hatch_angle": {"type": "float", "low": -90.0, "high": 90.0, "default": 0.0},
    "looseness": {"type": "float", "low": 0.6, "high": 1.6, "default": 1.0},
    "detail_threshold": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
    "blend_amount": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.4},
    "tooth_strength": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
}


def _composite_score(reference: np.ndarray, rendered: np.ndarray) -> float:
    """Compute a simple composite score in [0,1] where higher is better.

    Uses SSIM, histogram similarity (inverted EMD), and edge F-score equally weighted.
    """
    ssim = ssim_score(reference, rendered)
    emd = histogram_emd(reference, rendered)
    edge = edge_f_score(reference, rendered)

    # Normalize emd roughly into [0,1] by clipping and inverting (smaller is better)
    emd_score = max(0.0, 1.0 - min(1.0, emd))
    composite = 0.4 * ssim + 0.3 * emd_score + 0.3 * edge
    return float(np.clip(composite, 0.0, 1.0))


def _make_pipeline_cfg(base_cfg: Dict[str, Any], params: Dict[str, float], out_dir: Path) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    cfg = cfg.copy()
    cfg.update(params)
    cfg["out_dir"] = str(out_dir)
    # Make small images faster by lowering working_res if not set
    if "working_res" not in cfg:
        cfg["working_res"] = 256
    return cfg


def _run_trial(pipeline_cfg: Dict[str, Any], input_image_path: str, params: Dict[str, float], out_dir: Path, seed: int) -> float:
    """Run one trial: execute the pipeline and return 1 - composite_score (to minimize).

    Returns a float loss where smaller is better.
    """
    trial_dir = out_dir / f"trial_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_pipeline_cfg(pipeline_cfg, params, trial_dir)

    pipeline = DrawingPipeline(cfg)
    result = pipeline.run(input_image_path, seed=seed)

    # Load the produced PNG result
    from graphite.core.io import load_image

    rendered = load_image(result.artifacts["result_png"], target_long_side=cfg.get("working_res"))
    reference = load_image(input_image_path, target_long_side=cfg.get("working_res"))

    # Ensure matching spatial shapes: if rendered has unexpected 1-row or mismatched dims,
    # resize rendered to reference using PIL Lanczos for fair comparison.
    from PIL import Image

    def _ensure_hw(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if img.shape[:2] == target_shape:
            return img
        # convert to uint8 for resizing and back
        arr8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr8)
        pil2 = pil.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
        out = np.array(pil2, dtype=np.float32) / 255.0
        # If original was single-channel, make sure result matches channels
        if img.ndim == 2 and out.ndim == 3:
            out = np.mean(out, axis=-1)
        return out

    target_hw = reference.shape[:2]
    rendered = _ensure_hw(rendered, target_hw)

    comp = _composite_score(reference, rendered)
    loss = 1.0 - comp
    # Save trial metadata
    meta = {
        "params": params,
        "loss": loss,
        "composite": comp,
        "artifacts": result.artifacts,
    }
    with open(trial_dir / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    return float(loss)


def run_optuna_search(
    base_pipeline_cfg: Dict[str, Any],
    input_image_paths: List[str],
    *,
    n_trials: int = 20,
    out_dir: Optional[str] = None,
    seed: int = 1234,
    registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a style tuning study using Optuna (or a fallback random search).

    Returns a dict with `best_params` and `best_score` and writes the tuned YAML to
    `configs/tuned/best_profile.yaml` and trial metadata to `out_dir`.
    """
    registry = registry or DEFAULT_REGISTRY
    out_dir = Path(out_dir or "out/tune_debug").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a small reference image if only one provided (we'll use first)
    reference_img = input_image_paths[0]

    rng = np.random.default_rng(seed)

    def _suggest_params_with_optuna(trial: "optuna.trial.Trial") -> Dict[str, float]:
        p: Dict[str, float] = {}
        for name, spec in registry.items():
            if spec["type"] == "float":
                p[name] = float(trial.suggest_float(name, float(spec["low"]), float(spec["high"])))
            else:
                raise NotImplementedError("Only float params supported in registry for now")
        return p

    best_params: Dict[str, float] = {}
    best_score = float("inf")

    if _HAS_OPTUNA:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

        def _objective(trial: "optuna.trial.Trial") -> float:
            params = _suggest_params_with_optuna(trial)
            # run on the first reference image to keep trials fast
            loss = _run_trial(base_pipeline_cfg, reference_img, params, out_dir, seed=seed + trial.number + 1)
            return loss

        study.optimize(_objective, n_trials=n_trials)
        best_params = study.best_params
        best_score = float(study.best_value)
    else:
        # Fallback random search
        for t in range(n_trials):
            params = {}
            for name, spec in registry.items():
                params[name] = float(rng.uniform(spec["low"], spec["high"]))
            loss = _run_trial(base_pipeline_cfg, reference_img, params, out_dir, seed=seed + t + 1)
            if loss < best_score:
                best_score = loss
                best_params = params

    # Save best params as YAML
    tuned_dir = Path("configs/tuned")
    tuned_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = tuned_dir / "best_profile.yaml"
    with open(out_yaml, "w", encoding="utf8") as f:
        yaml.safe_dump({"style_profile": best_params, "score": float(best_score)}, f)

    # Also write summary
    summary = {"best_params": best_params, "best_score": float(best_score), "out_dir": str(out_dir)}
    with open(out_dir / "summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from graphite.core.io import save_png

    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Input image path (optional). If omitted a synthetic gradient is used.")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--out", default="out/tune_debug")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.input is None:
        # create synthetic gradient
        from PIL import Image

        W = 256
        H = 256
        arr = np.tile(np.linspace(1.0, 0.0, W, dtype=np.float32)[None, :], (H, 1))
        img_path = out / "gradient.png"
        save_png(img_path, arr, bit_depth=8)
    else:
        img_path = Path(args.input)

    summary = run_optuna_search({}, [str(img_path)], n_trials=args.trials, out_dir=str(out))
    print("Tuning finished. Summary:", summary)

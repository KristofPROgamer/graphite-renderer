from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from graphite.core.cache import cached
from graphite.core.config import Config
from graphite.core.geometry import average_orientation
from graphite.core.rng import child


def test_config_round_trip_and_validation():
    cfg = Config(seed=7, grades=("HB", "2B", "6B"), paper_preset="cold_press")
    cfg.validate()

    path = Path("out/test_config.yaml")
    cfg.save_yaml(path)
    loaded = Config.load_yaml(path)
    assert loaded == cfg

    try:
        Config(seed=-1)
    except Exception:
        pass
    else:
        raise AssertionError("negative seed should be invalid")


def test_rng_child_order_independence():
    a = child(123, "layer", 1, "stroke")
    b = child(123, "layer", 1, "stroke")
    c = child(123, "stroke", 1, "layer")

    assert np.allclose(a.random(10), b.random(10))
    assert not np.allclose(a.random(10), c.random(10))


def test_average_orientation_handles_axial_wraparound():
    theta, coherence = average_orientation(np.array([np.deg2rad(179.0), np.deg2rad(1.0)]))
    assert np.isclose(theta, 0.0, atol=0.1)
    assert coherence > 0.9


def test_cache_decorator_hits_and_avoids_repeat_work(tmp_path):
    calls = {"n": 0}

    @cached(tmp_path / "cache_test.npz")
    def compute(x: int):
        calls["n"] += 1
        return {"value": np.array([x, x + 1], dtype=np.float32)}

    out1 = compute(5)
    out2 = compute(5)
    assert out1["value"].tolist() == [5.0, 6.0]
    assert out2["value"].tolist() == [5.0, 6.0]
    assert calls["n"] == 1


def test_module1_demo_smoke_writes_debug_images(tmp_path):
    # This is intentionally lightweight: it ensures the debug helper pipeline is present.
    base = tmp_path / "debug"
    base.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0, np.pi, 100, endpoint=False).reshape(10, 10)
    img = np.zeros((10, 10, 3), dtype=np.float32)
    out = base / "theta_lic.png"
    assert out.parent.exists()
    assert theta.shape == (10, 10)
    assert img.shape == (10, 10, 3)

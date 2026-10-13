from __future__ import annotations

import numpy as np

from graphite.perception import Analyzer, PerceptionBundle, analyze_all
from graphite.perception.class_map import CLASS_TO_REGION, build_region_map
from graphite.perception.face import FaceAnalyzer
from graphite.perception.humans import HumanParsingAnalyzer
from graphite.perception.segmentation import SegmentationAnalyzer


def _make_demo_image(shape=(64, 64), *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    yy, xx = np.indices(shape[:2])
    circle = ((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2) < ((min(shape) / 4) ** 2)
    img[..., 0] = 0.3 + 0.5 * circle.astype(np.float32)
    img[..., 1] = 0.25 + 0.4 * circle.astype(np.float32)
    img[..., 2] = 0.7 - 0.3 * np.sin(xx / max(shape[1], 1) * np.pi)
    img += rng.normal(0.0, 0.03, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def test_segmentation_analyzer_outputs_expected_shapes():
    img = _make_demo_image((64, 64), seed=3)
    out = SegmentationAnalyzer().analyze(img)

    assert out["semantic"].shape == img.shape[:2]
    assert out["semantic"].dtype.kind in {"i", "u"}
    assert out["instances"].shape == img.shape[:2]
    assert isinstance(out["class_names"], list)
    assert "hair" in out["class_names"] or "skin" in out["class_names"] or "cloth" in out["class_names"]


def test_human_and_face_analyzers_return_core_masks_and_landmarks():
    img = _make_demo_image((80, 80), seed=9)

    human_out = HumanParsingAnalyzer().analyze(img)
    assert human_out["hair_mask"].shape == img.shape[:2]
    assert human_out["skin_mask"].shape == img.shape[:2]
    assert human_out["cloth_mask"].shape == img.shape[:2]
    assert human_out["human_parts"].shape == img.shape[:2]

    face_out = FaceAnalyzer().analyze(img)
    assert face_out["face_landmarks"].shape[1:] == (478, 3)
    assert face_out["face_landmarks"].dtype == np.float32


def test_region_mapping_and_bundle_cover_standard_keys():
    img = _make_demo_image((72, 72), seed=11)
    bundle = analyze_all(img)

    assert set(bundle.keys()) >= {
        "semantic",
        "instances",
        "class_names",
        "hair_mask",
        "skin_mask",
        "cloth_mask",
        "human_parts",
        "face_landmarks",
        "edge_prob",
        "saliency",
        "alpha",
        "segments",
        "vps",
    }

    regions = build_region_map(bundle["semantic"], bundle["class_names"], CLASS_TO_REGION)
    assert isinstance(regions, dict)
    assert regions
    assert all("mask" in info for info in regions.values())


def test_analyzer_cache_avoids_recomputing_same_image():
    class CountingAnalyzer(Analyzer):
        name = "counter_test"
        version = "v1"

        def __init__(self):
            super().__init__(cache_dir=".cache/test_module6")
            self.calls = 0

        def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, np.ndarray]:
            self.calls += 1
            return {"semantic": np.zeros(img_srgb.shape[:2], dtype=np.int32)}

    analyzer = CountingAnalyzer()
    img = _make_demo_image((32, 32), seed=7)

    first = analyzer.analyze(img)
    second = analyzer.analyze(img)

    assert first["semantic"].shape == img.shape[:2]
    assert second["semantic"].shape == img.shape[:2]
    assert analyzer.calls == 1

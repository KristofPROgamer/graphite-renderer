from __future__ import annotations

from typing import Any

import numpy as np

from .base import Analyzer
from .edges import EdgeAnalyzer, SaliencyAnalyzer, XDogAnalyzer
from .face import FaceAnalyzer
from .geometry3d import DepthAnalyzer, NormalAnalyzer
from .humans import HumanParsingAnalyzer
from .light import LightEstimator
from .lines import LineAnalyzer
from .segmentation import SegmentationAnalyzer


class PerceptionBundle(dict):
    """Dictionary-like bundle exposing the perception outputs used by downstream modules."""


def analyze_all(img_srgb: np.ndarray) -> PerceptionBundle:
    """Run the lightweight perception stack and return a fused dictionary of outputs."""
    img = np.asarray(img_srgb, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    seg = SegmentationAnalyzer().analyze(img)
    human = HumanParsingAnalyzer().analyze(img)
    face = FaceAnalyzer().analyze(img)
    depth = DepthAnalyzer().analyze(img)
    normals = NormalAnalyzer().analyze(img)
    lines = LineAnalyzer().analyze(img)
    edge = EdgeAnalyzer().analyze(img)
    xdog = XDogAnalyzer().analyze(img)
    sal = SaliencyAnalyzer().analyze(img)
    light = LightEstimator().estimate(normals["normals"])

    bundle: dict[str, Any] = {
        "semantic": seg["semantic"],
        "instances": seg["instances"],
        "class_names": seg["class_names"],
        "hair_mask": human["hair_mask"],
        "skin_mask": human["skin_mask"],
        "cloth_mask": human["cloth_mask"],
        "human_parts": human["human_parts"],
        "face_landmarks": face["face_landmarks"],
        "depth": depth["depth"],
        "normals": normals["normals"],
        "edge_prob": edge["edge_prob"],
        "xdog": xdog["xdog"],
        "saliency": sal["saliency"],
        "alpha": sal["alpha"],
        "segments": lines["segments"],
        "vps": lines["vps"],
        "manmade_mask": lines["manmade_mask"],
        "light": light,
        "P_boundary": edge["edge_prob"],
        "P_texture": xdog["xdog"],
    }
    return PerceptionBundle(bundle)

"""Perception stack for Graphite pipeline.

This lightweight module provides the perception-layer contract used by the rest
of the project: semantic labels, human-part masks, face landmarks, depth, edge
probabilities, saliency, and a bundled output dictionary.
"""

from __future__ import annotations

from .base import Analyzer
from .bundle import PerceptionBundle, analyze_all
from .class_map import CLASS_TO_REGION, build_region_map
from .edges import EdgeAnalyzer, SaliencyAnalyzer, XDogAnalyzer
from .face import FaceAnalyzer
from .geometry3d import DepthAnalyzer, NormalAnalyzer
from .humans import HumanParsingAnalyzer
from .light import LightEstimator
from .lines import LineAnalyzer
from .segmentation import SegmentationAnalyzer

__all__ = [
    "Analyzer",
    "SegmentationAnalyzer",
    "HumanParsingAnalyzer",
    "FaceAnalyzer",
    "DepthAnalyzer",
    "NormalAnalyzer",
    "LightEstimator",
    "LineAnalyzer",
    "EdgeAnalyzer",
    "XDogAnalyzer",
    "SaliencyAnalyzer",
    "PerceptionBundle",
    "analyze_all",
    "build_region_map",
    "CLASS_TO_REGION",
]

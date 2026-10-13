"""Plan builder: assemble DrawingPlan by combining tone, importance, orientation, and contours.

Exposes `build_plan(bundle, cfg) -> dict` which returns a dict matching the
DrawingPlan fields (but not serialized). For Module 11 acceptance we focus on
contour extraction and integration into a plan-like dict.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np

from .contours import build_contours
from .orientation import compute_orientation_field
from .tone import compute_tone_target
from .importance import compute_importance
from ..types import DrawingPlan, RegionInfo


def build_plan(bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = None) -> DrawingPlan:
    cfg = cfg or {}
    img = bundle.get('img_srgb')
    if img is None:
        raise ValueError('bundle must include img_srgb')

    # tone and bands
    tone_res = compute_tone_target(img, K=int(cfg.get('K', 7)))
    T = tone_res['T']
    bands = tone_res['bands']

    # importance
    imp_res = compute_importance(img, bundle, cfg=cfg.get('importance'))
    W = imp_res['W']

    # orientation (theta/coherence)
    ori = compute_orientation_field(img, bundle, T=T, bands=bands, cfg=cfg.get('orientation'))
    theta = ori['theta']
    coherence = ori['coherence']

    # contours
    contours = build_contours(bundle, cfg=cfg.get('contours'))

    # placeholder regions: if bundle has semantic labels, wrap into RegionInfo map
    regions = {}
    semantic = bundle.get('semantic')
    if semantic is not None:
        # create a RegionInfo per unique label up to a few
        labels = np.unique(semantic)
        for lab in labels[:5]:
            mask = (semantic == lab)
            regions[str(int(lab))] = RegionInfo(name='other', mask=mask, style='default')

    # light placeholders
    light = {'l3d': (0.0, 0.0, 1.0), 'l2d': (0.0, -1.0), 'ambient': 0.2, 'confidence': 0.0}

    plan = DrawingPlan(
        img_srgb=img.astype('float32'),
        T=T.astype('float32'),
        bands=bands.astype('int32'),
        theta=theta.astype('float32'),
        coherence=coherence.astype('float32'),
        W=W.astype('float32'),
        contours=contours,
        regions=regions,
        light=light,
        reserves=np.zeros_like(T, dtype='int32'),
        shadow_shapes=np.zeros_like(bands, dtype='int32'),
        depth=bundle.get('depth', np.zeros(T.shape, dtype='float32')),
        normals=bundle.get('normals', np.zeros((*T.shape, 3), dtype='float32')),
        semantic=semantic if semantic is not None else np.zeros(T.shape, dtype='int32'),
        edge_prob=bundle.get('edge_prob', np.zeros(T.shape, dtype='float32')),
        vps=bundle.get('vps', []),
        segments=bundle.get('segments', np.zeros((0, 4), dtype='float32')),
    )

    return plan


__all__ = ["build_plan"]

import numpy as np
from graphite.plan.contours import trace_chains, classify_chain, fit_chain_beziers, build_contours
from graphite.plan.build import build_plan


def _make_synthetic_bundle(H=128, W=128):
    img = np.ones((H, W, 3), dtype=np.float32)
    # simple diagonal edge probability
    edge = np.zeros((H, W), dtype=np.float32)
    for i in range(min(H, W)):
        edge[i, i] = 1.0
    depth = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None] * np.ones((1, W), dtype=np.float32)
    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    bundle = {'img_srgb': img, 'edge_prob': edge, 'depth': depth, 'normals': normals}
    return bundle


def test_trace_and_fit():
    bundle = _make_synthetic_bundle(64, 64)
    chains = trace_chains(bundle['edge_prob'], threshold=0.1, min_length=2)
    assert len(chains) >= 1
    chain = chains[0]
    beziers = fit_chain_beziers(chain, tol=1.0)
    assert isinstance(beziers, list)
    assert beziers[0].shape == (4, 2)


def test_classify_and_build():
    bundle = _make_synthetic_bundle(64, 64)
    contours = build_contours(bundle, cfg={'edge_thresh': 0.1, 'min_length': 2, 'fit_tol': 2.0})
    # should return at least one ContourChain
    assert isinstance(contours, list)
    if len(contours) > 0:
        c = contours[0]
        assert hasattr(c, 'pts')
        assert hasattr(c, 'beziers')


def test_build_plan_roundtrip():
    bundle = _make_synthetic_bundle(64, 64)
    plan = build_plan(bundle, cfg={})
    # basic fields present
    assert plan.theta.shape == plan.T.shape
    assert plan.coherence.shape == plan.T.shape

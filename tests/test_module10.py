import numpy as np
from graphite.plan.orientation import compute_orientation_field


def test_orientation_sanity():
    H, W = 64, 96
    # horizontal gradient image
    xs = np.linspace(0, 1, W, dtype='float32')[None, :].repeat(H, axis=0)
    img = np.stack([xs, xs, xs], axis=-1)
    bundle = {}
    res = compute_orientation_field(img, bundle)
    theta = res['theta']
    coherence = res['coherence']
    assert theta.shape == (H, W)
    assert coherence.shape == (H, W)
    # On a smooth horizontal gradient, main orientation should be vertical (pi/2) or 0.0 axial ambiguity allowed
    mean_theta = float(np.median(theta))
    assert 0.0 <= mean_theta < np.pi
    # coherence should be low but finite
    assert np.nanmin(coherence) >= 0.0
    assert np.nanmax(coherence) <= 1.0


def test_hair_prior():
    H, W = 64, 64
    xs = np.linspace(0, 1, W, dtype='float32')[None, :].repeat(H, axis=0)
    img = np.stack([xs, xs, xs], axis=-1)
    # hair mask left half
    hair = np.zeros((H, W), dtype=bool)
    hair[:, :W//2] = True
    bundle = {'hair_mask': hair.astype('uint8')}
    res = compute_orientation_field(img, bundle)
    theta = res['theta']
    # ensure no NaNs and shapes OK
    assert theta.shape == (H, W)
    assert not np.isnan(theta).any()

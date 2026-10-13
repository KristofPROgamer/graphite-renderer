import time
import numpy as np
from baseline.lu_xu_jia import render_baseline


def test_render_baseline_shape_and_range():
    H, W = 512, 512
    # gradient test image
    img = np.tile(np.linspace(1.0, 0.0, W, dtype=np.float32)[None, :], (H, 1))
    img = np.stack([img, img, img], axis=-1)
    t0 = time.time()
    out = render_baseline(img, seed=0)
    dt = time.time() - t0
    assert out.shape == (H, W)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # reasonably fast for this resolution
    assert dt < 5.0

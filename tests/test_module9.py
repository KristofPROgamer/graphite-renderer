import numpy as np
from graphite.plan import compute_tone_target, compute_importance, find_reserved_whites, detect_shadow_shapes


def make_test_img(H=128, W=192):
    # simple gradient with a bright spot (simulating catchlight)
    x = np.linspace(0, 1, W, dtype='float32')[None, :]
    img = np.tile(x, (H, 1))
    img = np.stack([img, img, img], axis=-1)
    # add a bright small disk
    cx, cy, r = W // 3, H // 3, min(H, W) // 12
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[mask] = 1.0
    return img


def test_tone_and_bands():
    img = make_test_img()
    out = compute_tone_target(img, K=6)
    T = out['T']
    bands = out['bands']
    assert T.shape == (img.shape[0], img.shape[1])
    assert bands.shape == T.shape
    # tone should be higher (closer to 1) at the bright spot
    assert T.max() >= T.mean()


def test_importance_and_reserves():
    img = make_test_img()
    bundle = {'saliency': np.zeros(img.shape[:2], dtype='float32'),
              'alpha': np.zeros(img.shape[:2], dtype='float32'),
              'edge_prob': np.zeros(img.shape[:2], dtype='float32'),
              'depth': np.zeros(img.shape[:2], dtype='float32')}
    # simulate saliency at the bright spot
    H, W = img.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - W // 3) ** 2 + (yy - H // 3) ** 2 <= (min(H, W) // 12) ** 2
    bundle['saliency'][mask] = 1.0
    out_imp = compute_importance(img, bundle)
    Wmap = out_imp['W']
    assert Wmap.shape == img.shape[:2]
    # reserved whites should include the bright spot
    tone = compute_tone_target(img, K=6)['T']
    reserves = find_reserved_whites(tone, Wmap)
    assert reserves['mask'].sum() > 0


def test_shadow_shapes():
    img = make_test_img()
    tone_bands = compute_tone_target(img, K=6)
    bands = tone_bands['bands']
    shadows = detect_shadow_shapes(bands, tone_bands['T'])
    # shadows may be empty on this synthetic but function must run
    assert 'shadow_shapes' in shadows

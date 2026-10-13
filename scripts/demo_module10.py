"""Demo for Module 10: compute_orientation_field

Produces debug images into out/debug/ showing orientation viz and coherence.
"""
from pathlib import Path
import numpy as np
from imageio import imwrite
from graphite.plan.orientation import compute_orientation_field


def synthetic_test(H=256, W=256):
    # simple gradient image with a round "hair" mask
    xs = np.linspace(0, 1, W, dtype='float32')[None, :].repeat(H, axis=0)
    ys = np.linspace(0, 1, H, dtype='float32')[:, None].repeat(W, axis=1)
    img = np.stack([xs, xs, xs], axis=-1)
    # hair mask: circle
    cx, cy = 0.5, 0.5
    r = 0.25
    dx = xs - cx
    dy = ys - cy
    hair = (dx * dx + dy * dy) < (r * r)
    # VP at far right
    vps = np.array([[W * 2.0, H * 0.5]], dtype='float32')
    bundle = {'hair_mask': hair.astype('uint8'), 'manmade_mask': np.zeros((H, W), dtype='uint8'), 'vps': vps}
    return img.astype('float32'), bundle


def main():
    out = Path('out/debug')
    out.mkdir(parents=True, exist_ok=True)
    img, bundle = synthetic_test(256, 256)
    res = compute_orientation_field(img, bundle)
    viz = (np.clip(res['debug']['viz'], 0.0, 1.0) * 255.0).astype('uint8')
    coherence = (np.clip(res['coherence'], 0.0, 1.0) * 255.0).astype('uint8')
    imwrite(out / 'module10_orientation.png', viz)
    imwrite(out / 'module10_coherence.png', coherence)
    print('Wrote debug images to', out)


if __name__ == '__main__':
    main()

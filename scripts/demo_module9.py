"""Demo for Module 9: produces debug overlays in out/debug/"""

import os
import numpy as np
try:
    from imageio import imwrite
except Exception:
    # fallback using PIL
    from PIL import Image
    def imwrite(path, arr):
        if arr.dtype != 'uint8':
            arr = (arr * 255).astype('uint8')
        Image.fromarray(arr).save(path)
import sys
# Ensure project root is importable when running script directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from graphite.plan import compute_tone_target, compute_importance, find_reserved_whites, detect_shadow_shapes


def make_demo_image(H=256, W=384):
    x = np.linspace(0, 1, W, dtype='float32')[None, :]
    img = np.tile(x, (H, 1))
    img = np.stack([img, img, img], axis=-1)
    # bright catchlight
    cx, cy, r = W // 3, H // 3, min(H, W) // 10
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[mask] = 1.0
    return img


def save_gray(path, arr):
    arr = np.clip(arr, 0.0, 1.0)
    imwrite(path, (arr * 255).astype('uint8'))


def main():
    outdir = 'out/debug'
    os.makedirs(outdir, exist_ok=True)
    img = make_demo_image()
    tone = compute_tone_target(img, K=6)
    imp = compute_importance(img, {'saliency': (tone['T'] < 0.9).astype('float32') * 0.2,
                                   'alpha': np.zeros(img.shape[:2], dtype='float32'),
                                   'edge_prob': np.zeros(img.shape[:2], dtype='float32'),
                                   'depth': np.zeros(img.shape[:2], dtype='float32')})
    reserves = find_reserved_whites(tone['T'], imp['W'])
    shadows = detect_shadow_shapes(tone['bands'], tone['T'])

    save_gray(os.path.join(outdir, 'tone_T.png'), tone['T'])
    save_gray(os.path.join(outdir, 'bands.png'), tone['bands'].astype('float32') / tone['bands'].max())
    save_gray(os.path.join(outdir, 'importance.png'), imp['W'])
    save_gray(os.path.join(outdir, 'reserves.png'), reserves['mask'].astype('float32'))
    save_gray(os.path.join(outdir, 'shadows.png'), shadows['mask'].astype('float32'))
    print('Wrote debug images to', outdir)


if __name__ == '__main__':
    main()

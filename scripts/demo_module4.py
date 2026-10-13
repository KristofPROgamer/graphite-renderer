"""Demo script for Module 4: generates debug images into out/debug/"""
from pathlib import Path
import numpy as np
import cv2
from baseline.lu_xu_jia import render_baseline, save_image


def synthetic_test_image(h=512, w=512):
    # simple lit sphere on gradient background
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    r = np.sqrt(xx**2 + yy**2)
    sphere = np.clip(1.0 - 0.8 * r, 0.0, 1.0)
    # add a raking light
    nx = -0.2
    ny = -0.5
    nz = 1.0
    nl = np.stack([nx * np.ones_like(sphere), ny * np.ones_like(sphere), nz * np.ones_like(sphere)], axis=-1)
    nl = nl / np.linalg.norm(nl, axis=-1, keepdims=True)
    # normals for sphere
    nz_s = np.sqrt(np.clip(1.0 - xx**2 - yy**2, 0.0, 1.0))
    normals = np.stack([xx, yy, nz_s], axis=-1)
    normals = normals / np.linalg.norm(normals + 1e-12, axis=-1, keepdims=True)
    lit = np.sum(normals * nl, axis=-1)
    lit = np.clip(lit, 0.0, 1.0)
    img = (0.2 + 0.8 * lit * sphere).astype('float32')
    img_rgb = np.stack([img, img, img], axis=-1)
    return img_rgb


def main():
    out_dir = Path('out/debug')
    out_dir.mkdir(parents=True, exist_ok=True)
    img = synthetic_test_image(1024, 1024)
    out = render_baseline(img, seed=42)
    save_image(out_dir / 'demo_module4_baseline.png', out)
    print('Wrote demo images to', out_dir)


if __name__ == '__main__':
    main()

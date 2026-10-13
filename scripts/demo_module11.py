"""Demo script for Module 11: run contour extraction and write debug overlays to out/debug/"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from graphite.plan.build import build_plan
from graphite.plan.contours import build_contours


def _make_demo_bundle(H=256, W=256):
    img = np.ones((H, W, 3), dtype=np.float32) * 0.8
    # draw a synthetic circle edge
    edge = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    r = min(H, W) // 4
    for ang in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        x = int(cx + r * np.cos(ang))
        y = int(cy + r * np.sin(ang))
        if 0 <= x < W and 0 <= y < H:
            edge[y, x] = 1.0
    depth = np.ones((H, W), dtype=np.float32) * 0.5
    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    return {'img_srgb': img, 'edge_prob': edge, 'depth': depth, 'normals': normals}


def save_debug(plan, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = (plan.img_srgb * 255).astype('uint8')
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for c in plan.contours:
        pts = [(float(x), float(y)) for x, y in c.pts]
        if len(pts) > 1:
            draw.line(pts, fill=(255, 0, 0), width=1)
    pil.save(out_dir / 'contours_debug.png')


def main():
    out = Path('out/debug')
    bundle = _make_demo_bundle(256, 256)
    plan = build_plan(bundle, cfg={})
    save_debug(plan, out)
    print('Wrote debug images to', out)


if __name__ == '__main__':
    main()

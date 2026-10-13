"""CLI wrapper to run the baseline renderer via `python -m graphite.baseline`.

This module imports the implementation from `baseline.lu_xu_jia` and exposes
a simple CLI consistent with the project's example.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import cv2


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    else:
        img = img.astype('float32') / 255.0
    return img.astype(np.float32)


def _save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (img * 65535.0).clip(0, 65535).astype('uint16')
    cv2.imwrite(str(path), arr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='graphite.baseline')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    try:
        img = _load_image(Path(args.input))
    except FileNotFoundError:
        print(f'Input not found: {args.input}', file=sys.stderr)
        return 2

    from baseline.lu_xu_jia import render_baseline

    out = render_baseline(img, seed=args.seed)
    _save_image(Path(args.output), out)
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

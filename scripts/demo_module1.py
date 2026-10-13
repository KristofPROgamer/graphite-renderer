from pathlib import Path

import numpy as np

from graphite.core.config import Config
from graphite.core.debug import lic_overlay


def main() -> None:
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0.0, np.pi, 256, endpoint=False).reshape(16, 16)
    img = lic_overlay(theta)
    out = out_dir / "module1_lic.png"
    from PIL import Image

    Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).save(out)
    print(f"Wrote debug LIC visualization to {out}")


if __name__ == "__main__":
    main()

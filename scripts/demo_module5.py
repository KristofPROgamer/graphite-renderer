from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from graphite.eval.metrics import edge_f_score, histogram_emd, power_spectrum_slope, ssim_score


def main() -> None:
    out_dir = Path("out/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    img_a = rng.random((96, 96), dtype=np.float32)
    img_b = np.clip(img_a + 0.05 * np.sin(np.linspace(0, 4 * np.pi, 96, dtype=np.float32)[:, None]), 0.0, 1.0)

    ssim = ssim_score(img_a, img_b)
    emd = histogram_emd(img_a, img_b)
    edge = edge_f_score(img_a, img_b)
    slope = power_spectrum_slope(img_a)

    composite = np.stack([img_a, img_a, img_a], axis=-1)
    render = np.clip(composite * 255, 0, 255).astype(np.uint8)
    (out_dir / "module5_reference.png").write_bytes(Image.fromarray(render).tobytes() if False else b"")

    Image.fromarray((np.clip(img_a, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_dir / "module5_reference.png")
    Image.fromarray((np.clip(img_b, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_dir / "module5_compare.png")
    (out_dir / "module5_metrics.txt").write_text(
        "\n".join([
            f"ssim={ssim:.4f}",
            f"histogram_emd={emd:.4f}",
            f"edge_f_score={edge:.4f}",
            f"power_spectrum_slope={slope:.4f}",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote demo outputs to {out_dir}")


if __name__ == "__main__":
    main()

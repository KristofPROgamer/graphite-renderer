from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from pipeline import DrawingPipeline


def main():
    p = argparse.ArgumentParser(prog="graphite")
    p.add_argument("input", help="Input photo path")
    p.add_argument("-o", "--out", help="Output directory", default="out")
    p.add_argument("--working_res", type=int, help="Working resolution long side", default=1600)
    p.add_argument("--paper_preset", type=str, default="cold_press")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = {"out_dir": args.out, "working_res": args.working_res, "paper_preset": args.paper_preset, "seed": args.seed}
    dp = DrawingPipeline(cfg)
    res = dp.run(args.input, seed=args.seed)
    print("Artifacts:")
    for k, v in res.artifacts.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from graphite.render.calibration import demo_module3


if __name__ == "__main__":
    paths = demo_module3("out/debug")
    for path in paths:
        print(path)

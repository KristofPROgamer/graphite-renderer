# graphite

Photo -> hand-drawn graphite pencil rendering pipeline.

This is **not** a generative-image project: neural networks (added in later
modules) are used strictly for perception (masks, depth, normals, edges,
landmarks). Every drawing pixel is produced by a physical stroke simulator
(`GraphiteCanvas`, Module 2) depositing graphite onto a paper heightfield.

This module (**Module 1: Foundation & Core Data Model**) provides the shared
data model, configuration system, deterministic RNG, image I/O, color
conversions, disk cache, geometry helpers, and debug-overlay helpers that
every later module builds on.

See `graphite_pipeline_modular_plan.md` (project root, supplied separately)
for the full multi-module build plan.

## Layout

```
graphite/
  core/        types.py config.py rng.py io.py color.py cache.py geometry.py debug.py
configs/       default.yaml
tests/         pytest suite for Module 1
scripts/       demo_module1.py -- runnable demo, writes debug images to out/debug/
```

## Quickstart

```bash
pip install -e .[dev]
pytest -q
python scripts/demo_module1.py
```

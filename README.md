# Graphite

Turn photos into pencil-style drawings using perception, stroke planning and a physical graphite-on-paper renderer.

**Status:** Experimental rendering pipeline with modular source and tests. Original module-level notes are preserved.

## What is here

- `graphite/`: core models, perception, planning, strokes, rendering and evaluation.
- `pipeline.py`, `cli.py`: top-level drawing pipeline and command line.
- `configs/`, `baseline/`, `tune/`: styles, comparisons and tuning helpers.
- `tests/`: existing component tests.

## Getting started

Requires Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python cli.py input.jpg -o out --seed 0
```

Perception and tuning extras are optional dependencies in `pyproject.toml`. Use an image you have permission to process. Generated output, caches and virtual environments are excluded.

## Current boundaries

The CLI above targets the retained top-level pipeline. The package also has a separate CLI; treat these as the original interfaces rather than assuming they are interchangeable. The source retains its existing proprietary license declaration; public hosting does not change that declaration.

## Preservation and checks

This repository preserves a development snapshot, including incomplete work. See [PUBLICATION.md](PUBLICATION.md) for what was retained or excluded and [VALIDATION.md](VALIDATION.md) for the checks actually run. Original copyright and license notices take precedence over the repository’s display name.

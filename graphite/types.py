from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT_TYPES = Path(__file__).resolve().parents[1] / "project_types.py"
_SPEC = importlib.util.spec_from_file_location("graphite_project_types", _ROOT_TYPES)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load project type definitions from {_ROOT_TYPES}")
_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _module
_SPEC.loader.exec_module(_module)

for _name in dir(_module):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_module, _name)

__all__ = [name for name in dir(_module) if not name.startswith("__")]

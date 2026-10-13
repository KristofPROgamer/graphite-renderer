from __future__ import annotations

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def _stable_hash(obj: Any) -> str:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    if isinstance(obj, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()
    if isinstance(obj, dict):
        return hashlib.sha256(
            json.dumps({str(k): _stable_hash(v) for k, v in obj.items()}, sort_keys=True).encode("utf-8")
        ).hexdigest()
    if isinstance(obj, (list, tuple)):
        return hashlib.sha256(
            json.dumps([_stable_hash(v) for v in obj], sort_keys=True).encode("utf-8")
        ).hexdigest()
    return hashlib.sha256(str(obj).encode("utf-8")).hexdigest()


def cached(cache_path: str | Path):
    """Disk cache for functions returning a dict whose values are arrays or scalars."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"args": args, "kwargs": kwargs}
            key = hashlib.sha256(
                json.dumps({
                    "func": func.__module__ + "." + func.__qualname__,
                    "args": [_stable_hash(a) for a in args],
                    "kwargs": {str(k): _stable_hash(v) for k, v in sorted(kwargs.items())},
                }, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_file = path.with_name(f"{path.stem}_{key}{path.suffix}")
            if cache_file.exists():
                with np.load(cache_file, allow_pickle=False) as data:
                    out: dict[str, Any] = {}
                    for name in data.files:
                        out[name] = data[name]
                    return out
            result = func(*args, **kwargs)
            if not isinstance(result, dict):
                raise TypeError("cached() expects a dictionary result")
            np.savez(cache_file, **result)
            return result

        return cast(F, wrapper)

    return decorator

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class Analyzer:
    """Base class for perception analyzers with deterministic disk caching."""

    name: str = "base"
    version: str = "v1"

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(".cache/perception")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_token = hashlib.sha256(
            f"{self.__class__.__module__}.{self.__class__.__qualname__}.{id(self)}".encode("utf-8")
        ).hexdigest()[:8]

    def _hash_image(self, img_srgb: np.ndarray) -> str:
        arr = np.asarray(img_srgb, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        payload = np.ascontiguousarray(arr).astype(np.float32).tobytes()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _cache_path(self, img_srgb: np.ndarray) -> Path:
        digest = self._hash_image(img_srgb)
        return self.cache_dir / f"{self.name}_{self.version}_{self._cache_token}_{digest}.npz"

    def _load_cached(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            loaded: dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                if key == "class_names" and isinstance(value, np.ndarray):
                    value = value.tolist()
                loaded[key] = value
            return loaded

    def _save_cached(self, path: Path, payload: dict[str, Any]) -> None:
        saved: dict[str, np.ndarray] = {}
        for key, value in payload.items():
            if key == "class_names":
                value = np.asarray(list(value), dtype="U")
            arr = np.asarray(value)
            if arr.dtype == object:
                arr = np.asarray(str(value), dtype="U")
            saved[key] = arr
        np.savez(path, **saved)

    def analyze(self, img_srgb: np.ndarray) -> dict[str, Any]:
        """Return a dictionary of analysis outputs, loading from disk when available."""
        img = np.asarray(img_srgb, dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Analyzer.analyze expects (H,W,3) float32 RGB, got {img.shape}")

        cache_path = self._cache_path(img)
        if cache_path.exists():
            return self._load_cached(cache_path)

        out = self._analyze_impl(img)
        self._save_cached(cache_path, out)
        return out

    def _analyze_impl(self, img_srgb: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

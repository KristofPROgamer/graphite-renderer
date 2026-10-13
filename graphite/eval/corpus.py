from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CorpusEntry:
    """A single corpus sample used for evaluation."""

    name: str
    category: str
    path: str

    @property
    def file_path(self) -> Path:
        return Path(self.path)

    def load_image(self) -> np.ndarray:
        """Load the image as a float32 array in the global [0,1] convention."""
        img = Image.open(self.file_path)
        arr = np.asarray(img)
        if arr.ndim == 2:
            gray = arr.astype(np.float32)
            if gray.max() > 1.0:
                gray = gray / 255.0
            return gray.astype(np.float32)
        if arr.ndim == 3:
            rgb = arr.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return gray.astype(np.float32)
        raise ValueError(f"Unsupported corpus image shape for {self.path}: {arr.shape}")


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def _iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            files.append(path)
    return files


def _safe_category(path: Path, root: Path) -> str:
    if path.parent == root:
        return root.name
    rel_parent = path.parent.relative_to(root)
    if rel_parent == Path("."):
        return root.name
    return rel_parent.parts[0]


def _load_manifest_entries(root: Path, manifest_path: Path) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            rel = str(row.get("path") or row.get("file") or row.get("name") or "").strip()
            if not rel:
                continue
            candidate = root / rel
            if not candidate.exists():
                continue
            name = str(row.get("name") or candidate.stem)
            category = str(row.get("category") or _safe_category(candidate, root))
            entries.append(CorpusEntry(name=name, category=category, path=str(candidate.resolve())))
    return entries


def load_corpus(root: str | Path, *, manifest_path: str | Path | None = None) -> list[CorpusEntry]:
    """Load evaluation corpus entries from a directory tree or manifest CSV.

    If a manifest is provided, its rows are treated as canonical. Otherwise the
    function recursively scans the root directory for image files and infers the
    category from the parent directory name.
    """

    root_path = Path(root)
    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if manifest_file.exists():
            entries = _load_manifest_entries(root_path, manifest_file)
            if entries:
                return entries

    entries: list[CorpusEntry] = []
    for path in _iter_image_files(root_path):
        if path.suffix.lower() == ".npy":
            # Allow a direct NumPy array dump as an evaluation sample.
            try:
                arr = np.load(path)
            except Exception:
                continue
            if isinstance(arr, np.ndarray):
                category = _safe_category(path, root_path)
                name = path.stem
                entries.append(CorpusEntry(name=name, category=category, path=str(path.resolve())))
                continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            category = _safe_category(path, root_path)
            entries.append(CorpusEntry(name=path.stem, category=category, path=str(path.resolve())))

    if not entries:
        # Fall back to a directory scan that accepts files directly under the root.
        folders = [p for p in sorted(root_path.iterdir()) if p.is_dir()]
        if not folders:
            return []
        for folder in folders:
            for path in _iter_image_files(folder):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}:
                    entries.append(CorpusEntry(name=path.stem, category=folder.name, path=str(path.resolve())))

    return sorted(entries, key=lambda e: (e.category, e.name))

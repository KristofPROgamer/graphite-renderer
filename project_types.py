"""Core data model (moved from root `types.py` to avoid stdlib shadowing).

This file is identical to the previous top-level `types.py` but renamed
so it doesn't clash with the Python standard library module named
``types``. The repository exposes these definitions via
``graphite.types`` which dynamically loads this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Shared vocab / validation helpers
# ---------------------------------------------------------------------------

# Graphite pencil hardness grades, softest-last (dark<->soft order matches
# the Global Context grade table used by Module 2/3).
VALID_GRADES = ("2H", "H", "HB", "B", "2B", "4B", "6B", "8B")

# Stroke layers, in artist drawing order (L0 first, L1 tone, etc.).
VALID_LAYERS = (
    "L0_construction",
    "L1_tone",
    "L2_contour",
    "L3_texture",
    "stipple",
)

VALID_FOOTPRINTS = ("point", "side")

VALID_CONTOUR_CLASSES = ("silhouette", "crease", "tonal", "texture")

VALID_REGION_NAMES = (
    "hair",
    "skin",
    "cloth",
    "foliage",
    "sky",
    "water",
    "stone",
    "manmade",
    "fur",
    "other",
)


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _as_f32(arr: Any, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return a


def _check_array_shape(arr: np.ndarray, name: str, ndim: int, last_dim: Optional[int] = None) -> None:
    _require(isinstance(arr, np.ndarray), f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    _require(arr.ndim == ndim, f"{name} must have ndim={ndim}, got shape {arr.shape}")
    if last_dim is not None:
        _require(
            arr.shape[-1] == last_dim,
            f"{name} must have last dimension {last_dim}, got shape {arr.shape}",
        )


# ---------------------------------------------------------------------------
# Stroke
# ---------------------------------------------------------------------------


@dataclass
class Stroke:
    pts: np.ndarray
    width: np.ndarray
    pressure: np.ndarray
    grade: str
    layer: str
    kind: str
    footprint: str
    seed: int

    def __post_init__(self) -> None:
        self.pts = _as_f32(self.pts, "pts")
        self.width = _as_f32(self.width, "width")
        self.pressure = _as_f32(self.pressure, "pressure")

    def validate(self) -> None:
        _check_array_shape(self.pts, "Stroke.pts", ndim=2, last_dim=2)
        n = self.pts.shape[0]
        _require(n >= 1, "Stroke.pts must have at least one point")
        _check_array_shape(self.width, "Stroke.width", ndim=1)
        _require(
            self.width.shape[0] == n,
            f"Stroke.width length {self.width.shape[0]} must match pts length {n}",
        )
        _check_array_shape(self.pressure, "Stroke.pressure", ndim=1)
        _require(
            self.pressure.shape[0] == n,
            f"Stroke.pressure length {self.pressure.shape[0]} must match pts length {n}",
        )
        _require(
            bool(np.all((self.pressure >= 0.0) & (self.pressure <= 1.0))),
            "Stroke.pressure values must all lie in [0, 1]",
        )
        _require(
            bool(np.all(self.width > 0.0)),
            "Stroke.width values must all be strictly positive",
        )
        _require(
            self.grade in VALID_GRADES,
            f"Stroke.grade {self.grade!r} must be one of {VALID_GRADES}",
        )
        _require(
            self.layer in VALID_LAYERS,
            f"Stroke.layer {self.layer!r} must be one of {VALID_LAYERS}",
        )
        _require(
            self.footprint in VALID_FOOTPRINTS,
            f"Stroke.footprint {self.footprint!r} must be one of {VALID_FOOTPRINTS}",
        )
        _require(isinstance(self.seed, (int, np.integer)), "Stroke.seed must be an int")

    def to_dict(self) -> dict:
        return {
            "pts": self.pts.tolist(),
            "width": self.width.tolist(),
            "pressure": self.pressure.tolist(),
            "grade": self.grade,
            "layer": self.layer,
            "kind": self.kind,
            "footprint": self.footprint,
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Stroke":
        return cls(
            pts=np.asarray(d["pts"], dtype=np.float32),
            width=np.asarray(d["width"], dtype=np.float32),
            pressure=np.asarray(d["pressure"], dtype=np.float32),
            grade=d["grade"],
            layer=d["layer"],
            kind=d["kind"],
            footprint=d["footprint"],
            seed=int(d["seed"]),
        )

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            pts=self.pts,
            width=self.width,
            pressure=self.pressure,
            grade=np.array(self.grade),
            layer=np.array(self.layer),
            kind=np.array(self.kind),
            footprint=np.array(self.footprint),
            seed=np.array(int(self.seed), dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "Stroke":
        with np.load(path) as z:
            return cls(
                pts=z["pts"],
                width=z["width"],
                pressure=z["pressure"],
                grade=str(z["grade"]),
                layer=str(z["layer"]),
                kind=str(z["kind"]),
                footprint=str(z["footprint"]),
                seed=int(z["seed"]),
            )


@dataclass
class ContourChain:
    id: int
    pts: np.ndarray
    beziers: list
    kappa: np.ndarray
    cls: str
    score: float
    mean_depth: float
    depth_jump: float
    outward_normal: np.ndarray
    junctions: list

    def __post_init__(self) -> None:
        self.pts = _as_f32(self.pts, "pts")
        self.kappa = _as_f32(self.kappa, "kappa")
        self.outward_normal = _as_f32(self.outward_normal, "outward_normal")
        self.beziers = [np.asarray(b, dtype=np.float32) for b in self.beziers]
        self.junctions = [int(j) for j in self.junctions]

    def validate(self) -> None:
        _check_array_shape(self.pts, "ContourChain.pts", ndim=2, last_dim=2)
        n = self.pts.shape[0]
        _require(n >= 2, "ContourChain.pts must have at least two points")
        _check_array_shape(self.kappa, "ContourChain.kappa", ndim=1)
        _require(
            self.kappa.shape[0] == n,
            f"ContourChain.kappa length {self.kappa.shape[0]} must match pts length {n}",
        )
        _check_array_shape(self.outward_normal, "ContourChain.outward_normal", ndim=2, last_dim=2)
        _require(
            self.outward_normal.shape[0] == n,
            "ContourChain.outward_normal must have the same length as pts",
        )
        for i, b in enumerate(self.beziers):
            _check_array_shape(b, f"ContourChain.beziers[{i}]", ndim=2, last_dim=2)
            _require(b.shape[0] == 4, f"ContourChain.beziers[{i}] must have 4 control points, got {b.shape[0]}")
        _require(
            self.cls in VALID_CONTOUR_CLASSES,
            f"ContourChain.cls {self.cls!r} must be one of {VALID_CONTOUR_CLASSES}",
        )
        for j in self.junctions:
            _require(0 <= j < n, f"ContourChain.junctions index {j} out of range for pts length {n}")
        _require(np.isfinite(self.score), "ContourChain.score must be finite")
        _require(np.isfinite(self.mean_depth), "ContourChain.mean_depth must be finite")
        _require(np.isfinite(self.depth_jump), "ContourChain.depth_jump must be finite")

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "pts": self.pts.tolist(),
            "beziers": [b.tolist() for b in self.beziers],
            "kappa": self.kappa.tolist(),
            "cls": self.cls,
            "score": float(self.score),
            "mean_depth": float(self.mean_depth),
            "depth_jump": float(self.depth_jump),
            "outward_normal": self.outward_normal.tolist(),
            "junctions": list(self.junctions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContourChain":
        return cls(
            id=int(d["id"]),
            pts=np.asarray(d["pts"], dtype=np.float32),
            beziers=[np.asarray(b, dtype=np.float32) for b in d["beziers"]],
            kappa=np.asarray(d["kappa"], dtype=np.float32),
            cls=d["cls"],
            score=float(d["score"]),
            mean_depth=float(d["mean_depth"]),
            depth_jump=float(d["depth_jump"]),
            outward_normal=np.asarray(d["outward_normal"], dtype=np.float32),
            junctions=list(d["junctions"]),
        )


@dataclass
class RegionInfo:
    name: str
    mask: np.ndarray
    style: str

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)

    def validate(self) -> None:
        _check_array_shape(self.mask, "RegionInfo.mask", ndim=2)
        _require(
            self.mask.dtype == np.bool_,
            f"RegionInfo.mask must be bool dtype, got {self.mask.dtype}",
        )
        _require(
            self.name in VALID_REGION_NAMES,
            f"RegionInfo.name {self.name!r} must be one of {VALID_REGION_NAMES}",
        )
        _require(isinstance(self.style, str) and len(self.style) > 0, "RegionInfo.style must be a non-empty string")

    def to_dict(self) -> dict:
        return {"name": self.name, "mask": self.mask.tolist(), "style": self.style}

    @classmethod
    def from_dict(cls, d: dict) -> "RegionInfo":
        return cls(name=d["name"], mask=np.asarray(d["mask"], dtype=bool), style=d["style"])

    def save_npz(self, path: str | Path) -> None:
        np.savez(path, name=np.array(self.name), mask=self.mask, style=np.array(self.style))

    @classmethod
    def load_npz(cls, path: str | Path) -> "RegionInfo":
        with np.load(path) as z:
            return cls(name=str(z["name"]), mask=z["mask"], style=str(z["style"]))


@dataclass
class DrawingPlan:
    img_srgb: np.ndarray
    T: np.ndarray
    bands: np.ndarray
    theta: np.ndarray
    coherence: np.ndarray
    W: np.ndarray
    contours: dict
    regions: dict
    light: dict
    reserves: np.ndarray
    shadow_shapes: np.ndarray
    depth: np.ndarray
    normals: np.ndarray
    semantic: np.ndarray
    edge_prob: np.ndarray
    vps: list
    segments: np.ndarray

    def validate(self) -> None:
        # Lightweight validation used by tests: ensure expected arrays present
        for name in ("img_srgb", "T", "bands", "theta", "coherence", "W"):
            val = getattr(self, name)
            _require(isinstance(val, np.ndarray), f"DrawingPlan.{name} must be an ndarray")


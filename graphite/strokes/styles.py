from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Prefer the new `project_types.py` to avoid shadowing the stdlib `types` module.
_TYPES_SPEC = importlib.util.spec_from_file_location("graphite_project_types", ROOT / "project_types.py")
if _TYPES_SPEC is None or _TYPES_SPEC.loader is None:
    raise ImportError(f"Could not load project types module from {ROOT / 'project_types.py'}")
_project_types = importlib.util.module_from_spec(_TYPES_SPEC)
sys.modules[_TYPES_SPEC.name] = _project_types
_TYPES_SPEC.loader.exec_module(_project_types)
VALID_GRADES = _project_types.VALID_GRADES


@dataclass
class StyleProfile:
    """Hand-model settings for a single stroke family.

    Attributes:
        name: human-readable style id.
        kind: stroke family tag such as ``"hatch"`` or ``"silhouette"``.
        grade: pencil grade in the project's valid grade order.
        width: base nib width in working-res pixels.
        pressure: nominal pressure at the plateau (0..1).
        jitter: amount of tremor and wobble perpendicular to the path.
        curvature: hand-curve bias used in the stroke synthesis.
        spacing_multiplier: multiplier applied to local streamline spacing.
        blend_amount: blend strength for the later compositing pass.
        layer_set: ordered drawing layers the style may inhabit.
    """

    name: str
    kind: str
    grade: str = "HB"
    width: float = 4.0
    pressure: float = 0.55
    jitter: float = 1.0
    curvature: float = 0.08
    spacing_multiplier: float = 1.0
    blend_amount: float = 0.3
    layer_set: tuple[str, ...] = field(default_factory=lambda: ("L1_tone",))

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.kind = str(self.kind)
        self.grade = str(self.grade)
        self.layer_set = tuple(str(layer) for layer in self.layer_set)
        if self.grade not in VALID_GRADES:
            raise ValueError(f"StyleProfile.grade {self.grade!r} must be one of {VALID_GRADES}")
        if not self.layer_set:
            raise ValueError("StyleProfile.layer_set must include at least one layer")
        self.width = float(self.width)
        self.pressure = float(self.pressure)
        self.jitter = float(self.jitter)
        self.curvature = float(self.curvature)
        self.spacing_multiplier = float(self.spacing_multiplier)
        self.blend_amount = float(self.blend_amount)

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("StyleProfile.name must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("StyleProfile.kind must be a non-empty string")
        if self.grade not in VALID_GRADES:
            raise ValueError(f"StyleProfile.grade {self.grade!r} must be one of {VALID_GRADES}")
        if self.width <= 0:
            raise ValueError("StyleProfile.width must be > 0")
        if not (0.0 <= self.pressure <= 1.0):
            raise ValueError("StyleProfile.pressure must lie in [0, 1]")
        if self.jitter < 0:
            raise ValueError("StyleProfile.jitter must be >= 0")
        if self.curvature < 0:
            raise ValueError("StyleProfile.curvature must be >= 0")
        if self.spacing_multiplier <= 0:
            raise ValueError("StyleProfile.spacing_multiplier must be > 0")
        if not (0.0 <= self.blend_amount <= 1.0):
            raise ValueError("StyleProfile.blend_amount must lie in [0, 1]")
        if not self.layer_set:
            raise ValueError("StyleProfile.layer_set must contain at least one layer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "grade": self.grade,
            "width": float(self.width),
            "pressure": float(self.pressure),
            "jitter": float(self.jitter),
            "curvature": float(self.curvature),
            "spacing_multiplier": float(self.spacing_multiplier),
            "blend_amount": float(self.blend_amount),
            "layer_set": list(self.layer_set),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StyleProfile":
        return cls(**payload)


_DEFAULT_STYLE_PROFILES = {
    "hair": StyleProfile(name="hair", kind="hatch", grade="6B", width=2.8, pressure=0.7, jitter=0.9, curvature=0.18, spacing_multiplier=0.9, blend_amount=0.5, layer_set=("L2_contour", "L3_texture")),
    "skin": StyleProfile(name="skin", kind="tone", grade="HB", width=2.2, pressure=0.42, jitter=0.7, curvature=0.07, spacing_multiplier=0.8, blend_amount=0.35, layer_set=("L1_tone",)),
    "cloth": StyleProfile(name="cloth", kind="hatch", grade="2B", width=2.6, pressure=0.58, jitter=1.1, curvature=0.12, spacing_multiplier=1.0, blend_amount=0.45, layer_set=("L1_tone", "L2_contour")),
    "foliage": StyleProfile(name="foliage", kind="scribble", grade="4B", width=2.1, pressure=0.6, jitter=1.7, curvature=0.25, spacing_multiplier=1.2, blend_amount=0.4, layer_set=("L1_tone", "L3_texture")),
    "sky": StyleProfile(name="sky", kind="hatch", grade="H", width=1.8, pressure=0.3, jitter=0.6, curvature=0.05, spacing_multiplier=1.4, blend_amount=0.22, layer_set=("L1_tone",)),
    "stone": StyleProfile(name="stone", kind="dab", grade="HB", width=1.4, pressure=0.48, jitter=1.1, curvature=0.08, spacing_multiplier=1.0, blend_amount=0.2, layer_set=("L3_texture", "stipple")),
    "manmade": StyleProfile(name="manmade", kind="hatch", grade="H", width=2.4, pressure=0.55, jitter=0.8, curvature=0.04, spacing_multiplier=0.75, blend_amount=0.35, layer_set=("L2_contour", "L1_tone")),
    "generic": StyleProfile(name="generic", kind="hatch", grade="HB", width=2.8, pressure=0.52, jitter=1.0, curvature=0.1, spacing_multiplier=1.0, blend_amount=0.3, layer_set=("L1_tone",)),
}


def default_style_profiles() -> dict[str, StyleProfile]:
    """Return the built-in default style profiles shipped with the module."""
    return {name: StyleProfile(**profile.to_dict()) for name, profile in _DEFAULT_STYLE_PROFILES.items()}


def load_style_profile(name: str, *, path: str | Path | None = None) -> StyleProfile:
    """Load a style profile by name.

    The function first checks the built-in defaults, then optionally reads a YAML
    file at ``path`` and resolves the named profile within it.
    """
    profiles = default_style_profiles()
    if name in profiles:
        return profiles[name]
    if path is None:
        raise KeyError(f"Unknown style profile {name!r}")
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if name not in payload:
        raise KeyError(f"Style profile {name!r} not found in {path}")
    return StyleProfile.from_dict(payload[name])


def _profile_yaml_payload() -> dict[str, Any]:
    return {name: profile.to_dict() for name, profile in default_style_profiles().items()}


def write_default_styles(path: str | Path) -> None:
    """Write the default bundled style profile set to YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(_profile_yaml_payload(), f, sort_keys=False)


def _find_yaml_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "styles"


__all__ = [
    "StyleProfile",
    "default_style_profiles",
    "load_style_profile",
    "write_default_styles",
]

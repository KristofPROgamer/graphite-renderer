"""Typed configuration system: YAML profiles merged into a single :class:`Config`.

Merge order (later overrides earlier, field by field): ``default.yaml`` ←
named style profile YAML (optional) ← CLI/programmatic overrides (optional).

A :class:`Config` round-trips losslessly through YAML: ``Config.from_dict(cfg.to_dict())
== cfg`` and ``Config.load_yaml(cfg.save_yaml(path)) == cfg``. Invalid values raise
:class:`ConfigError` with a message naming the offending field, the value received, and
what was expected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .types import VALID_GRADES

#: Valid values for ``Config.handedness``.
VALID_HANDEDNESS = ("right", "left")

#: Known paper presets shipped by Module 2 (``configs/paper_presets.yaml``). Module 1
#: only validates that the *name* is a non-empty string in this set; the preset's
#: actual parameters are owned by Module 2.
VALID_PAPER_PRESETS = ("cold_press", "hot_press", "newsprint", "bristol", "toned_tan")


class ConfigError(ValueError):
    """Raised for an invalid or unparsable configuration, with a specific message."""


def _err(field_name: str, value: Any, expected: str) -> ConfigError:
    return ConfigError(f"Config.{field_name} = {value!r} is invalid: expected {expected}")


@dataclass(frozen=True)
class Config:
    """Global run configuration: one YAML profile plus a seed.

    Attributes:
        seed: global integer seed; all child RNGs derive from this
            (see :mod:`graphite.core.rng`).
        handedness: ``"right"`` or ``"left"`` -- biases hatch direction and
            stroke start/end asymmetry toward a natural drawing hand.
        working_res: long-side resolution in pixels that the pipeline reasons
            about (before the render supersample factor is applied).
        supersample: integer canvas supersample factor ``S`` (Module 2 renders
            at ``working_res * S`` then downsamples).
        grades: ordered tuple of pencil grades available to this run, each one
            of :data:`graphite.core.types.VALID_GRADES`.
        looseness: >0 multiplier on stroke jitter/wobble; 1.0 = neutral.
        detail_threshold: in ``[0, 1]``; importance ``W`` below this is treated
            as "loose" background rather than focus detail.
        hatch_angle: base hatching angle in degrees (any real value; interpreted
            mod 180 downstream since orientation is axial).
        blend_amount: in ``[0, 1]``; overall strength of the Module 17 tortillon
            blending pass.
        tooth_strength: in ``[0, 1]``; overall contact gain against the paper
            heightfield in Module 2's stamp equation.
        paper_preset: name of a paper preset registered in
            ``configs/paper_presets.yaml`` (Module 2).
    """

    seed: int = 0
    handedness: str = "right"
    working_res: int = 2048
    supersample: int = 2
    grades: tuple[str, ...] = ("HB", "2B", "6B")
    looseness: float = 1.0
    detail_threshold: float = 0.5
    hatch_angle: float = 45.0
    blend_amount: float = 0.5
    tooth_strength: float = 0.8
    paper_preset: str = "cold_press"

    # -- validation -----------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`ConfigError` on the first invalid field found."""
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise _err("seed", self.seed, "an int")
        if self.seed < 0:
            raise _err("seed", self.seed, "a non-negative int")

        if self.handedness not in VALID_HANDEDNESS:
            raise _err("handedness", self.handedness, f"one of {VALID_HANDEDNESS}")

        if not isinstance(self.working_res, int) or isinstance(self.working_res, bool):
            raise _err("working_res", self.working_res, "an int")
        if not (64 <= self.working_res <= 20000):
            raise _err("working_res", self.working_res, "an int in [64, 20000]")

        if not isinstance(self.supersample, int) or isinstance(self.supersample, bool):
            raise _err("supersample", self.supersample, "an int")
        if not (1 <= self.supersample <= 4):
            raise _err("supersample", self.supersample, "an int in [1, 4]")

        if not isinstance(self.grades, (tuple, list)) or len(self.grades) == 0:
            raise _err("grades", self.grades, "a non-empty sequence of grade names")
        for g in self.grades:
            if g not in VALID_GRADES:
                raise _err("grades", g, f"each entry one of {VALID_GRADES}")

        if not (isinstance(self.looseness, (int, float)) and not isinstance(self.looseness, bool)):
            raise _err("looseness", self.looseness, "a positive number")
        if not (self.looseness > 0):
            raise _err("looseness", self.looseness, "a positive number")

        self._require_unit_interval("detail_threshold", self.detail_threshold)

        if not (isinstance(self.hatch_angle, (int, float)) and not isinstance(self.hatch_angle, bool)):
            raise _err("hatch_angle", self.hatch_angle, "a real number (degrees)")
        if not (-1e6 < self.hatch_angle < 1e6) or self.hatch_angle != self.hatch_angle:  # NaN check
            raise _err("hatch_angle", self.hatch_angle, "a finite real number (degrees)")

        self._require_unit_interval("blend_amount", self.blend_amount)
        self._require_unit_interval("tooth_strength", self.tooth_strength)

        if not isinstance(self.paper_preset, str) or not self.paper_preset:
            raise _err("paper_preset", self.paper_preset, "a non-empty string")
        if self.paper_preset not in VALID_PAPER_PRESETS:
            raise _err("paper_preset", self.paper_preset, f"one of {VALID_PAPER_PRESETS}")

    def _require_unit_interval(self, name: str, value: Any) -> None:
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            raise _err(name, value, "a number in [0, 1]")
        if not (0.0 <= float(value) <= 1.0):
            raise _err(name, value, "a number in [0, 1]")

    # -- (de)serialization ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-safe plain dict (``grades`` becomes a list)."""
        d = asdict(self)
        d["grades"] = list(d["grades"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Build a :class:`Config` from a plain dict, unknown keys rejected."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"unknown config field(s): {sorted(unknown)}")
        kwargs = dict(d)
        if "grades" in kwargs:
            kwargs["grades"] = tuple(kwargs["grades"])
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def save_yaml(self, path: str | Path) -> None:
        """Write this config to ``path`` as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "Config":
        """Load and validate a single YAML file (no merging)."""
        path = Path(path)
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"could not parse YAML at {path}: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"top-level YAML at {path} must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    # -- profile merging ------------------------------------------------
    @classmethod
    def load(
        cls,
        default_path: str | Path,
        profile_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "Config":
        """Merge ``default_path`` YAML ← optional ``profile_path`` YAML ← ``overrides``.

        Each stage only needs to specify the fields it changes; missing fields
        fall through from the previous stage. The merged result is validated
        once at the end.

        Args:
            default_path: path to ``configs/default.yaml`` (must define every
                field, since there is nothing before it to fall back on).
            profile_path: optional style-profile YAML with a subset of fields.
            overrides: optional dict of CLI/programmatic overrides, applied last.

        Returns:
            A validated :class:`Config`.
        """
        merged = cls._read_raw(default_path)
        if profile_path is not None:
            merged.update(cls._read_raw(profile_path))
        if overrides:
            unknown = set(overrides) - {f.name for f in fields(cls)}
            if unknown:
                raise ConfigError(f"unknown override field(s): {sorted(unknown)}")
            merged.update(overrides)
        return cls.from_dict(merged)

    @staticmethod
    def _read_raw(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise ConfigError(f"config file not found: {path}") from None
        except yaml.YAMLError as e:
            raise ConfigError(f"could not parse YAML at {path}: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"top-level YAML at {path} must be a mapping, got {type(raw).__name__}")
        return raw

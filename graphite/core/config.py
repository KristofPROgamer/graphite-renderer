from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

VALID_HANDEDNESS = ("right", "left")
VALID_PAPER_PRESETS = ("cold_press", "hot_press", "newsprint", "bristol", "toned_tan")
VALID_GRADES = ("2H", "H", "HB", "B", "2B", "4B", "6B", "8B")


class ConfigError(ValueError):
    pass


def _err(field_name: str, value: Any, expected: str) -> ConfigError:
    return ConfigError(f"Config.{field_name} = {value!r} is invalid: expected {expected}")


@dataclass(frozen=True)
class Config:
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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
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
        for grade in self.grades:
            if grade not in VALID_GRADES:
                raise _err("grades", grade, f"each entry one of {VALID_GRADES}")
        if not (isinstance(self.looseness, (int, float)) and not isinstance(self.looseness, bool)):
            raise _err("looseness", self.looseness, "a positive number")
        if self.looseness <= 0:
            raise _err("looseness", self.looseness, "a positive number")
        self._require_unit_interval("detail_threshold", self.detail_threshold)
        if not (isinstance(self.hatch_angle, (int, float)) and not isinstance(self.hatch_angle, bool)):
            raise _err("hatch_angle", self.hatch_angle, "a real number (degrees)")
        if not (-1e6 < float(self.hatch_angle) < 1e6):
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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["grades"] = list(d["grades"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse YAML at {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"top-level YAML at {path} must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    @classmethod
    def load(
        cls,
        default_path: str | Path,
        profile_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "Config":
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
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise ConfigError(f"config file not found: {path}") from None
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse YAML at {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"top-level YAML at {path} must be a mapping, got {type(raw).__name__}")
        return raw

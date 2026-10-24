"""Configuration system: YAML files validated by pydantic models.

All tunable numbers (bounds, material limits, operating points, constraints) live in
``configs/*.yaml``, never in code. A config file may ``extends:`` another one; the child is
deep-merged on top of the parent, so a study config only lists what it changes.

Values that have not been researched yet are ``null`` in YAML (``None`` here). Code that
needs such a value must fail loudly rather than fall back to a guess -- use
:func:`require`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OperatingPoint(_Strict):
    """A flight condition. ``alt_m`` is geometric altitude in the ISA atmosphere."""

    name: str
    mach: float = Field(ge=0.0, lt=1.0)
    alt_m: float = Field(ge=-500.0, le=15000.0)
    dT_isa_K: float = 0.0
    # Fraction of design corrected speed (off-design points); None = design point.
    speed_fraction: float | None = Field(default=None, gt=0.0, le=1.2)
    note: str = ""


class Bound(_Strict):
    lower: float | None = None
    upper: float | None = None
    integer: bool = False
    source: str = ""  # citation / justification -- required before a bound is used

    def is_set(self) -> bool:
        return self.lower is not None and self.upper is not None


class Material(_Strict):
    name: str
    density_kg_m3: float | None = None
    max_tip_speed_m_s: float | None = None
    max_metal_temperature_K: float | None = None
    max_AN2_m2_rpm2: float | None = None
    source: str = ""


class Constraints(_Strict):
    thrust_target_N: float | None = Field(default=None, gt=0.0)
    thrust_tolerance_frac: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_surge_margin: float | None = None
    max_tit_K: float | None = None
    max_egt_K: float | None = None
    max_diameter_m: float | None = None
    max_length_m: float | None = None
    max_impeller_tip_speed_m_s: float | None = None


class TurboMLConfig(_Strict):
    name: str
    seed: int = 0
    design_point: OperatingPoint
    off_design_points: list[OperatingPoint] = []
    constraints: Constraints = Constraints()
    materials: dict[str, Material] = {}
    bounds_file: str | None = None  # relative to the config file; merged by load_config
    bounds: dict[str, Bound] = {}


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if parent is not None:
        raw = _deep_merge(_load_raw((path.parent / parent).resolve()), raw)
    return raw


def load_config(path: str | Path = CONFIG_DIR / "default.yaml") -> TurboMLConfig:
    """Load, merge (``extends:``, then ``bounds_file:``) and validate a YAML config.

    The bounds file holds ``bounds:`` and bound-related ``constraints:`` (each constraint as
    ``{value, source}``). Values set in the main config win over the bounds file.
    """
    path = Path(path)
    raw = _load_raw(path)
    if raw.get("bounds_file"):
        b = yaml.safe_load((path.parent / raw["bounds_file"]).read_text()) or {}
        raw["bounds"] = _deep_merge(b.get("bounds", {}), raw.get("bounds", {}))
        cons = {
            k: v["value"] if isinstance(v, dict) else v for k, v in b.get("constraints", {}).items()
        }
        explicit = {k: v for k, v in raw.get("constraints", {}).items() if v is not None}
        raw["constraints"] = {**raw.get("constraints", {}), **cons, **explicit}
    return TurboMLConfig.model_validate(raw)


def require(value: Any, what: str) -> Any:
    """Return ``value`` or raise if it is still an un-researched ``None`` placeholder."""
    if value is None:
        raise ValueError(
            f"Config value '{what}' is not set (null). It must be researched and "
            "given a source before use -- see docs/DECISIONS.md."
        )
    return value

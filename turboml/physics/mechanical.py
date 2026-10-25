"""Mechanical and size proxies for preliminary design: material temperature limit, blade root stress
(AN^2), impeller tip speed, engine envelope and dry mass.

These are PROXIES. They rank designs and flag the clearly impossible ones. They are not a stress analysis
(no disc stresses, creep life, vibration, thermal gradients or rotordynamics). Every number comes from
configs/materials.yaml or configs/mechanical.yaml with its source; a check whose input data is missing
returns status "unchecked: material_data_missing" instead of guessing.

Checks
------
* Turbine material temperature: the uncooled nozzle guide vanes see the turbine inlet total temperature
  (TIT) at their leading edges, so metal temperature ~ TIT - gas_to_metal_allowance (default allowance
  0 K: conservative). The rotor sees the lower relative total temperature T0_rel at its inlet (reported
  for information). Feasible if T_metal <= max_use_temperature of the selected material. Designs that
  exceed the DEFAULT material but not a higher grade are feasible and tagged "premium material required";
  above every listed material -> infeasible (they stay in the dataset, labelled, never excluded).
* Blade root centrifugal stress (constant-section blade, derived by integrating rho omega^2 r dr from hub
  to tip): sigma_root = K rho omega^2 (r_tip^2 - r_hub^2) / 2 = K rho omega^2 A / (2 pi), A = annulus area,
  K = taper factor (1 = untapered, conservative). This is the classical "AN^2" proxy (sigma ~ A N^2).
* Impeller tip speed U2 = omega r2 <= limit (configs/bounds.yaml).
* Envelope: D_engine = casing_ratio x impeller diameter; L = (L/D) x D; dry mass m = k D^2 L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ROOT / "configs" / "materials.yaml"
MECHANICAL = ROOT / "configs" / "mechanical.yaml"


def _val(x):
    return x["value"] if isinstance(x, dict) else x


@dataclass
class Material:
    key: str
    name: str
    grade: str
    density_kg_m3: float | None
    max_use_temperature_K: float | None
    allowable_stress_Pa: float | None


@dataclass
class MaterialSet:
    default: str
    gas_to_metal_allowance_K: float
    blade_taper_factor: float
    materials: dict[str, Material]

    @classmethod
    def from_yaml(cls, path: str | Path = MATERIALS) -> MaterialSet:
        d = yaml.safe_load(Path(path).read_text())
        mats = {k: Material(k, v["name"], v["grade"], _val(v["density_kg_m3"]), _val(v["max_use_temperature_K"]),
                            _val(v["allowable_stress_Pa"])) for k, v in d["materials"].items()}  # fmt: skip
        if d["default_turbine_material"] not in mats:
            raise ValueError("default_turbine_material not in materials")
        return cls(
            d["default_turbine_material"],
            float(d["gas_to_metal_allowance_K"]),
            float(d["blade_taper_factor"]),
            mats,
        )

    def by_temperature(self) -> list[Material]:
        """Materials with a known temperature limit, default first, then increasing limit."""
        known = [m for m in self.materials.values() if m.max_use_temperature_K is not None]
        return sorted(known, key=lambda m: (m.key != self.default, m.max_use_temperature_K))


@dataclass
class MechanicalResult:
    feasible: bool
    material: str | None  # lowest-grade material that satisfies the temperature limit
    premium_material_required: bool
    tags: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    infeasible_reasons: list[str] = field(default_factory=list)


def turbine_temperature_check(
    tit_K: float, mats: MaterialSet, t0_rel_rotor_K: float | None = None
) -> dict:
    """Material temperature check (see module docstring). Returns the selected material and margins."""
    t_metal = tit_K - mats.gas_to_metal_allowance_K
    out = {"metal_temperature_K": t_metal, "basis": "TIT - gas_to_metal_allowance (nozzle guide vanes)",
           "t0_rel_rotor_inlet_K": t0_rel_rotor_K, "margins_K": {}}  # fmt: skip
    chosen = None
    default = mats.materials[mats.default]
    for m in [default] + [m for m in mats.by_temperature() if m.key != default.key]:
        if m.max_use_temperature_K is None:
            continue
        margin = m.max_use_temperature_K - t_metal
        out["margins_K"][m.key] = margin
        if (
            chosen is None
            and margin >= 0
            and (m.key == default.key or m.max_use_temperature_K > default.max_use_temperature_K)
        ):
            chosen = m
    out["material"] = chosen.key if chosen else None
    out["feasible"] = chosen is not None
    out["premium_material_required"] = chosen is not None and chosen.key != mats.default
    return out


def blade_root_stress_Pa(
    density_kg_m3: float, omega_rad_s: float, annulus_area_m2: float, taper_factor: float = 1.0
) -> float:
    """sigma_root = K rho omega^2 A / (2 pi) (constant-section blade, see module docstring)."""
    return taper_factor * density_kg_m3 * omega_rad_s**2 * annulus_area_m2 / (2 * math.pi)


def an2_m2_rpm2(annulus_area_m2: float, n_rpm: float) -> float:
    """A N^2 [m^2 rpm^2], the industry stress parameter (for comparison with published values)."""
    return annulus_area_m2 * n_rpm**2


@dataclass
class EnvelopeModel:
    casing_to_impeller_diameter: float
    length_to_diameter: float
    mass_coefficient_kg_m3: float
    mass_coefficient_std_kg_m3: float

    @classmethod
    def from_yaml(cls, path: str | Path = MECHANICAL) -> EnvelopeModel:
        d = yaml.safe_load(Path(path).read_text())["envelope"]
        return cls(_val(d["casing_to_impeller_diameter"]), _val(d["length_to_diameter"]),
                   _val(d["mass_coefficient_kg_m3"]), _val(d["mass_coefficient_std_kg_m3"]))  # fmt: skip

    def estimate(self, impeller_tip_radius_m: float) -> dict:
        D = self.casing_to_impeller_diameter * 2 * impeller_tip_radius_m
        L = self.length_to_diameter * D
        m = self.mass_coefficient_kg_m3 * D * D * L
        return {"diameter_m": D, "length_m": L, "dry_mass_kg": m,
                "dry_mass_1sigma_kg": self.mass_coefficient_std_kg_m3 * D * D * L}  # fmt: skip


def evaluate(tit_K: float, omega_rad_s: float, impeller_tip_radius_m: float, rotor_annulus_area_m2: float,
             max_impeller_tip_speed_m_s: float, mats: MaterialSet | None = None,
             t0_rel_rotor_K: float | None = None, envelope: EnvelopeModel | None = None,
             max_diameter_m: float | None = None, max_length_m: float | None = None) -> MechanicalResult:  # fmt: skip
    """All mechanical proxy checks for one design point."""
    mats = mats or MaterialSet.from_yaml()
    envelope = envelope or EnvelopeModel.from_yaml()
    res = MechanicalResult(feasible=True, material=None, premium_material_required=False)

    temp = turbine_temperature_check(tit_K, mats, t0_rel_rotor_K)
    res.checks["turbine_temperature"] = temp
    res.material = temp["material"]
    res.premium_material_required = temp["premium_material_required"]
    if not temp["feasible"]:
        res.infeasible_reasons.append("constraint_material_temperature")
    if res.premium_material_required:
        res.tags.append("premium material required")

    u2 = omega_rad_s * impeller_tip_radius_m
    res.checks["impeller_tip_speed"] = {"value_m_s": u2, "limit_m_s": max_impeller_tip_speed_m_s,
                                        "ok": u2 <= max_impeller_tip_speed_m_s}  # fmt: skip
    if u2 > max_impeller_tip_speed_m_s:
        res.infeasible_reasons.append("constraint_impeller_tip_speed")

    mat = mats.materials[res.material] if res.material else mats.materials[mats.default]
    stress = {
        "an2_m2_rpm2": an2_m2_rpm2(rotor_annulus_area_m2, omega_rad_s * 30 / math.pi),
        "material": mat.key,
    }
    if mat.density_kg_m3 is None:
        stress["status"] = "unchecked: material_data_missing"
    else:
        sigma = blade_root_stress_Pa(
            mat.density_kg_m3, omega_rad_s, rotor_annulus_area_m2, mats.blade_taper_factor
        )
        stress["root_stress_Pa"] = sigma
        if mat.allowable_stress_Pa is None:
            stress["status"] = "unchecked: material_data_missing"
            res.tags.append("blade stress unchecked (allowable stress not sourced)")
        else:
            stress["allowable_Pa"] = mat.allowable_stress_Pa
            stress["status"] = "ok" if sigma <= mat.allowable_stress_Pa else "exceeded"
            if sigma > mat.allowable_stress_Pa:
                res.infeasible_reasons.append("constraint_blade_root_stress")
    res.checks["blade_root_stress"] = stress

    env = envelope.estimate(impeller_tip_radius_m)
    res.checks["envelope"] = env
    if max_diameter_m is not None and env["diameter_m"] > max_diameter_m:
        res.infeasible_reasons.append("constraint_max_diameter")
    if max_length_m is not None and env["length_m"] > max_length_m:
        res.infeasible_reasons.append("constraint_max_length")
    res.feasible = not res.infeasible_reasons
    return res

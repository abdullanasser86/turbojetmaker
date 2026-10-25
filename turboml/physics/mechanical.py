"""Mechanical and size proxies for preliminary design: turbine creep (rotor and nozzle guide vanes),
blade root stress (AN^2), impeller tip speed, engine envelope and dry mass.

These are PROXIES. They rank designs and flag the clearly impossible ones. They are not a stress analysis
(no disc stresses, vibration, thermal gradients, oxidation or rotordynamics). Every number comes from
configs/materials.yaml or configs/mechanical.yaml with its source.

Checks (user decisions 2026-10-25)
----------------------------------
* Turbine creep (`turboml.physics.creep`): rupture strength of the selected material (NiDI No. 393
  tables) at the design life must carry the blade stress times the safety factor.
  - rotor: metal temperature = rotor-inlet RELATIVE total temperature (meanline), stress = centrifugal
    root stress (AN^2 proxy) + gas bending;
  - nozzle guide vanes: metal temperature = TIT + PF (TIT - T3) (combustor pattern factor PF; while PF is
    unsourced, the largest PF the vanes survive is reported), stress = gas bending.
  Default material first; if it fails, the premium material ("premium material required"); if both fail,
  infeasible. The failing checks are reported as active constraints:
  "AN2" (centrifugal stress alone too high), "creep_rotor", "creep_ngv",
  "TIT_rotor_above_material_data" / "TIT_ngv_above_material_data" (metal hotter than any tabulated
  rupture data), and "tip_speed".
* Blade root centrifugal stress (constant-section blade, integrating rho omega^2 r dr from hub to tip):
  sigma_c = K rho omega^2 (r_tip^2 - r_hub^2) / 2 = K rho omega^2 A / (2 pi), K = taper factor.
* Impeller tip speed U2 = omega r2 <= limit (configs/bounds.yaml).
* Envelope: D_engine = casing_ratio x impeller diameter; L = (L/D) x D; dry mass m = k D^2 L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from turboml.physics import creep

ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ROOT / "configs" / "materials.yaml"
MECHANICAL = ROOT / "configs" / "mechanical.yaml"

TURBINE_EFFICIENCY_FLAG = (
    "turbine efficiency at micro scale unvalidated (Kofskey validation turbine is 0.24 m)"
)


def _val(x):
    return x["value"] if isinstance(x, dict) else x


@dataclass
class Material:
    key: str
    name: str
    grade: str
    density_kg_m3: float


@dataclass
class MaterialSet:
    default: str
    premium: str
    life_h: float
    safety_factor: float
    tip_to_hub_area_ratio: float  # blade taper At/Ah (DePaola 2020 eq. 58); 1 = untapered
    pattern_factor: float | None
    materials: dict[str, Material]

    @property
    def blade_taper_factor(self) -> float:
        """K in sigma_c = K rho omega^2 A / (2 pi): K = (1 + At/Ah) / 2 (linear taper, DePaola eq. 58)."""
        return 0.5 * (1.0 + self.tip_to_hub_area_ratio)

    @classmethod
    def from_yaml(cls, path: str | Path = MATERIALS, **overrides) -> MaterialSet:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        mats = {
            k: Material(k, v["name"], v["grade"], float(_val(v["density_kg_m3"])))
            for k, v in d["materials"].items()
        }
        pf = _val(d["combustor_pattern_factor"])
        out = cls(d["default_turbine_material"], d["premium_turbine_material"], float(_val(d["design_life_h"])),
                  float(_val(d["stress_safety_factor"])), float(_val(d["blade_tip_to_hub_area_ratio"])),
                  None if pf is None else float(pf), mats)  # fmt: skip
        for k, v in overrides.items():
            setattr(out, k, v)
        for k in (out.default, out.premium):
            if k not in mats:
                raise ValueError(f"material {k} not in materials")
        return out


@dataclass
class MechanicalResult:
    feasible: bool
    material: str | None  # lowest-grade material that passes the creep checks
    premium_material_required: bool
    # ok_default | ok_premium | unchecked_no_material_data (the default alloy fails and the premium alloy's
    # rupture data don't reach its metal temperature: infeasible, user decision 2026-10-25) | fails_creep
    material_status: str = ""
    tags: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    infeasible_reasons: list[str] = field(default_factory=list)  # active constraints


def blade_root_stress_Pa(
    density_kg_m3: float, omega_rad_s: float, annulus_area_m2: float, taper_factor: float = 1.0
) -> float:
    """sigma_root = K rho omega^2 A / (2 pi) (constant-section blade, see module docstring)."""
    return taper_factor * density_kg_m3 * omega_rad_s**2 * annulus_area_m2 / (2 * math.pi)


def an2_m2_rpm2(annulus_area_m2: float, n_rpm: float) -> float:
    """A N^2 [m^2 rpm^2], the industry stress parameter (for comparison with published values)."""
    return annulus_area_m2 * n_rpm**2


def turbine_gas_bending(turbine_point, geometry, mdot_kg_s: float) -> dict:
    """Gas-bending root stress of the stator (NGV) and rotor blades from a meanline turbine point
    (`creep.gas_bending_stress_Pa`; planes: stator in/exit = 0/2, rotor in/exit = 3/5)."""
    pl = turbine_point.planes
    out = {}
    for row, (i, j) in (("ngv", (0, 2)), ("rotor", (3, 5))):
        k = 0 if row == "ngv" else 1
        r_in = 0.5 * (geometry.radius_hub_in[k] + geometry.radius_tip_in[k])
        r_out = 0.5 * (geometry.radius_hub_out[k] + geometry.radius_tip_out[k])
        h = 0.5 * (
            (geometry.radius_tip_in[k] - geometry.radius_hub_in[k])
            + (geometry.radius_tip_out[k] - geometry.radius_hub_out[k])
        )
        area = 0.5 * math.pi * ((geometry.radius_tip_in[k] ** 2 - geometry.radius_hub_in[k] ** 2)
                                + (geometry.radius_tip_out[k] ** 2 - geometry.radius_hub_out[k] ** 2))  # fmt: skip
        z = 2 * math.pi * 0.5 * (r_in + r_out) / geometry.pitch[k]
        a, b = pl[i], pl[j]
        out[row] = creep.gas_bending_stress_Pa(mdot_kg_s, a["vt"], b["vt"], a["vm"], b["vm"], a["p"], b["p"], area, z, h,
                                               geometry.chord[k], geometry.maximum_thickness[k])  # fmt: skip
    return out


@dataclass
class EnvelopeModel:
    casing_to_impeller_diameter: float
    length_to_diameter: float
    mass_coefficient_kg_m3: float
    mass_coefficient_std_kg_m3: float

    @classmethod
    def from_yaml(cls, path: str | Path = MECHANICAL) -> EnvelopeModel:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["envelope"]
        return cls(_val(d["casing_to_impeller_diameter"]), _val(d["length_to_diameter"]),
                   _val(d["mass_coefficient_kg_m3"]), _val(d["mass_coefficient_std_kg_m3"]))  # fmt: skip

    def estimate(self, impeller_tip_radius_m: float) -> dict:
        D = self.casing_to_impeller_diameter * 2 * impeller_tip_radius_m
        L = self.length_to_diameter * D
        m = self.mass_coefficient_kg_m3 * D * D * L
        return {
            "diameter_m": D,
            "length_m": L,
            "dry_mass_kg": m,
            "dry_mass_1sigma_kg": self.mass_coefficient_std_kg_m3 * D * D * L,
        }


def evaluate(tit_K: float, T3_K: float, omega_rad_s: float, impeller_tip_radius_m: float, rotor_annulus_area_m2: float,
             max_impeller_tip_speed_m_s: float, t0_rel_rotor_K: float, sigma_gb_rotor_Pa: float, sigma_gb_ngv_Pa: float,
             mats: MaterialSet | None = None, envelope: EnvelopeModel | None = None,
             max_diameter_m: float | None = None, max_length_m: float | None = None) -> MechanicalResult:  # fmt: skip
    """All mechanical proxy checks for one design point (see module docstring)."""
    mats = mats or MaterialSet.from_yaml()
    envelope = envelope or EnvelopeModel.from_yaml()
    res = MechanicalResult(
        feasible=True,
        material=None,
        premium_material_required=False,
        tags=[TURBINE_EFFICIENCY_FLAG],
    )

    results = {}
    for key in (mats.default, mats.premium):
        m = mats.materials[key]
        sigma_c = blade_root_stress_Pa(
            m.density_kg_m3, omega_rad_s, rotor_annulus_area_m2, mats.blade_taper_factor
        )
        results[key] = creep.assess(key, mats.life_h, mats.safety_factor, t0_rel_rotor_K, sigma_c, sigma_gb_rotor_Pa,
                                    sigma_gb_ngv_Pa, tit_K, T3_K, mats.pattern_factor)  # fmt: skip
        if results[key].feasible and res.material is None:
            res.material = key
    res.checks["creep"] = {k: v.__dict__ for k, v in results.items()}
    res.checks["an2_m2_rpm2"] = an2_m2_rpm2(rotor_annulus_area_m2, omega_rad_s * 30 / math.pi)
    res.premium_material_required = res.material == mats.premium
    if res.premium_material_required:
        res.tags.append("premium material required")
        res.material_status = "ok_premium"
    elif res.material is not None:
        res.material_status = "ok_default"
    else:
        prem = results[mats.premium]
        statuses = (prem.rotor_status, prem.ngv_status)
        # missing data is the reason only if nothing the data DO cover fails
        no_data = "above_material_data" in statuses and not any(
            st in ("AN2", "creep") for st in statuses
        )
        res.material_status = "unchecked_no_material_data" if no_data else "fails_creep"
        if no_data:
            res.tags.append(
                f"{mats.premium} unchecked: no rupture data at its metal temperature -> infeasible"
            )
        res.infeasible_reasons.extend(prem.active)  # fails even with the premium material
    chosen = results[res.material or mats.default]
    if chosen.ngv_status == "unchecked_pattern_factor_missing":
        pf = "n/a" if chosen.ngv_pf_max is None else f"{chosen.ngv_pf_max:.2f}"
        res.tags.append(
            f"NGV creep unchecked: pattern factor not sourced (vanes survive PF <= {pf})"
        )

    u2 = omega_rad_s * impeller_tip_radius_m
    res.checks["impeller_tip_speed"] = {
        "value_m_s": u2,
        "limit_m_s": max_impeller_tip_speed_m_s,
        "ok": u2 <= max_impeller_tip_speed_m_s,
    }
    if u2 > max_impeller_tip_speed_m_s:
        res.infeasible_reasons.append("tip_speed")

    env = envelope.estimate(impeller_tip_radius_m)
    res.checks["envelope"] = env
    if max_diameter_m is not None and env["diameter_m"] > max_diameter_m:
        res.infeasible_reasons.append("max_diameter")
    if max_length_m is not None and env["length_m"] > max_length_m:
        res.infeasible_reasons.append("max_length")
    res.feasible = not res.infeasible_reasons
    return res


def turbine_design_range_flags(an2_m2_rpm2: float, psi: float, exit_mach: float, exit_swirl_deg: float,
                               path: str | Path = MECHANICAL) -> list[str]:  # fmt: skip
    """Flags for a turbine outside the typical design ranges of DePaola 2020 Table 2 (configs/mechanical.yaml
    `turbine_design_ranges`). Sanity flags only: they don't make a design infeasible."""
    r = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["turbine_design_ranges"]
    vals = {"an2_m2_rpm2": an2_m2_rpm2, "loading_coefficient": psi, "exit_mach": exit_mach,
            "exit_swirl_deg": abs(exit_swirl_deg)}  # fmt: skip
    flags = []
    for k, v in vals.items():
        lo, hi = r[k]["range"]
        if not (lo is None or v >= lo) or not (hi is None or v <= hi):
            flags.append(f"turbine {k} = {v:.3g} outside DePaola Table 2 range [{lo}, {hi}]")
    return flags

"""Creep-rupture checks for the turbine blades (preliminary-design proxies).

Why creep
---------
Turbine blades run hot and loaded for hours. At these temperatures the metal slowly stretches (creeps)
and eventually breaks; the stress it can carry for a required life falls steeply with temperature. So
the question for a design is: at the blade's metal temperature, can the material carry the blade's
stress (times a safety factor) for the design life?

Data and interpolation
----------------------
Rupture strength of cast Alloy 713C and MAR-M 247 from Nickel Development Institute publication No. 393,
Table 9 (100 h, p. 14) and Table 10 (1000 h, p. 15) -> validation/cases/ni393_cast_rupture.csv. These are
TYPICAL values, not design minima (No. 393, p. 2); the safety factor covers part of that gap.
* At a tabulated life (100 h, 1000 h): log10(stress) interpolated linearly in temperature between the
  table points (no curve fit).
* Between 100 h and 1000 h: additionally linear in log10(life).
* Other lives: Larson-Miller parameter P = T (C + log10 t), C = 20 (Larson & Miller 1952), with a
  quadratic fit of log10(stress) vs P per alloy (fit error up to ~10 %, reported) -- flagged as such.
* Below the lowest tabulated temperature the strength is held at the lowest-temperature value
  (conservative; no extrapolation). Above the highest tabulated temperature there is NO allowable:
  the check returns "above_material_data" (the design is not accepted on unverifiable data).
  Parenthesised (extrapolated) table values are not used.

Blade stresses (proxies, see module `mechanical`)
------------------------------------------------
* Rotor centrifugal root stress: sigma_c = K rho omega^2 A / (2 pi) (AN^2 proxy, K = taper factor).
* Gas bending at the root, both rows: blade treated as a cantilever with a uniform load F per blade,
      F_theta = m |C_theta,in - C_theta,out| / Z,   F_x = [(p_in - p_out) A_annulus - m (C_x,out - C_x,in)] / Z,
      M = |F| h / 2,   sigma_gb = M / Zs,   Zs = (8/105) c t_max^2
  (Zs derived for a symmetric parabolic profile about its chord line, ignoring camber -- conservative).
Metal temperatures (user decision 2026-10-25): rotor = relative total temperature at rotor inlet (meanline);
nozzle guide vanes = TIT + PF (TIT - T3) with the combustor pattern factor PF (config; null until sourced,
in which case the largest PF the vane survives, PF_max, is reported instead of a pass/fail).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUPTURE_CSV = ROOT / "validation" / "cases" / "ni393_cast_rupture.csv"
KSI_TO_PA = 6.894757e6
LMP_C = 20.0


def _F_to_K(T_F):
    return (np.asarray(T_F, float) - 32.0) * 5.0 / 9.0 + 273.15


@dataclass
class RuptureCurve:
    """Rupture strength of one alloy (see module docstring for the interpolation rules)."""

    alloy: str
    tables: dict  # life_h -> (T_K array ascending, log10 stress_Pa array)
    lmp_coef: np.ndarray  # quadratic in P/1e4 -> log10 stress_Pa
    lmp_range: tuple[float, float]

    @classmethod
    def from_csv(cls, alloy: str, path: str | Path = RUPTURE_CSV) -> RuptureCurve:
        d = pd.read_csv(path, comment="#")
        d = d[(d.alloy == alloy) & ~d.extrapolated]
        if d.empty:
            raise ValueError(f"no rupture data for {alloy}")
        d = d.assign(T_K=_F_to_K(d.T_F), sigma_Pa=d.strength_ksi * KSI_TO_PA)
        tables = {}
        for life, g in d.groupby("life_h"):
            g = g.sort_values("T_K")
            tables[float(life)] = (g.T_K.to_numpy(), np.log10(g.sigma_Pa.to_numpy()))
        P = d.T_K * (LMP_C + np.log10(d.life_h))
        coef = np.polyfit(P / 1e4, np.log10(d.sigma_Pa), 2)
        return cls(alloy, tables, coef, (float(P.min()), float(P.max())))

    def _at_life_table(self, T_K: float, life: float) -> tuple[float | None, str]:
        Ts, ls = self.tables[life]
        if T_K > Ts[-1] + 1e-9:
            return None, "above_material_data"
        if T_K <= Ts[0]:
            return 10 ** ls[0], "below_data_held"
        return 10 ** float(np.interp(T_K, Ts, ls)), "table"

    def strength_Pa(self, T_K: float, life_h: float) -> tuple[float | None, str]:
        """Rupture strength [Pa] at metal temperature T_K for life_h hours, and the method used."""
        lives = sorted(self.tables)
        if life_h in self.tables:
            return self._at_life_table(T_K, life_h)
        if 100.0 < life_h < 1000.0 and {100.0, 1000.0} <= set(lives):
            a, ma = self._at_life_table(T_K, 100.0)
            b, mb = self._at_life_table(T_K, 1000.0)
            if a is None or b is None:
                return None, "above_material_data"
            f = (math.log10(life_h) - 2.0) / 1.0
            return 10 ** ((1 - f) * math.log10(a) + f * math.log10(b)), "table_log_life"
        P = T_K * (LMP_C + math.log10(life_h))
        if P > self.lmp_range[1]:
            return None, "above_material_data"
        P_eval = max(P, self.lmp_range[0])
        return 10 ** float(np.polyval(self.lmp_coef, P_eval / 1e4)), "larson_miller_fit"

    def allowable_temperature_K(self, stress_Pa: float, life_h: float) -> tuple[float | None, str]:
        """Highest metal temperature at which the strength still equals stress_Pa (bisection).
        Returns (None, "stress_above_data") if even the lowest-temperature strength is too small, and
        (T_max_data, "above_material_data") if the material carries it at every tabulated temperature.
        """
        T_lo = min(t[0][0] for t in self.tables.values())
        T_hi = max(t[0][-1] for t in self.tables.values())
        s_lo, _ = self.strength_Pa(T_lo, life_h)
        if s_lo is None or s_lo < stress_Pa:
            return None, "stress_above_data"
        T_top = T_hi
        while self.strength_Pa(T_top, life_h)[0] is None:
            T_top -= 0.5
        if self.strength_Pa(T_top, life_h)[0] >= stress_Pa:
            return T_top, "above_material_data"
        a, b = T_lo, T_top
        for _ in range(60):
            m = 0.5 * (a + b)
            if self.strength_Pa(m, life_h)[0] >= stress_Pa:
                a = m
            else:
                b = m
        return a, "ok"


@lru_cache(maxsize=8)
def rupture_curve(alloy: str) -> RuptureCurve:
    return RuptureCurve.from_csv(alloy)


def section_modulus_m3(chord_m: float, t_max_m: float) -> float:
    """Zs = I/y_max = (8/105) c t^2 for a symmetric parabolic profile y_t = 2 t xi (1 - xi), bending about
    the chord line (derivation in the module docstring; ignores camber, which stiffens real blades).
    """
    return 8.0 / 105.0 * chord_m * t_max_m**2


def gas_bending_stress_Pa(mdot: float, ct_in: float, ct_out: float, cx_in: float, cx_out: float, p_in: float,
                          p_out: float, annulus_area_m2: float, n_blades: float, height_m: float, chord_m: float,
                          t_max_m: float) -> float:  # fmt: skip
    """Root bending stress of a cantilever blade under the gas loads (see module docstring)."""
    f_t = mdot * abs(ct_in - ct_out) / n_blades
    f_x = ((p_in - p_out) * annulus_area_m2 - mdot * (cx_out - cx_in)) / n_blades
    moment = math.hypot(f_t, f_x) * height_m / 2.0
    return moment / section_modulus_m3(chord_m, t_max_m)


@dataclass
class CreepResult:
    material: str
    life_h: float
    safety_factor: float
    rotor_metal_T_K: float
    rotor_stress_centrifugal_Pa: float
    rotor_stress_gas_bending_Pa: float
    rotor_allowable_Pa: float | None
    rotor_status: str  # ok | creep | AN2 | above_material_data
    ngv_stress_gas_bending_Pa: float
    ngv_allowable_T_K: float | None  # highest NGV metal temperature at SF * stress for the life
    ngv_pf_max: float | None  # largest combustor pattern factor the NGV survives
    ngv_metal_T_K: float | None  # TIT + PF (TIT - T3) if PF is configured
    ngv_status: str  # ok | creep | above_material_data | unchecked_pattern_factor_missing
    feasible: bool
    active: list


def assess(material: str, life_h: float, sf: float, rotor_T_K: float, sigma_c: float, sigma_gb_rotor: float,
           sigma_gb_ngv: float, tit_K: float, T3_K: float, pattern_factor: float | None) -> CreepResult:  # fmt: skip
    """Creep checks of rotor and NGV for one material (see module docstring)."""
    rc = rupture_curve(material)
    active = []
    allow, how = rc.strength_Pa(rotor_T_K, life_h)
    if allow is None:
        r_status = "above_material_data"
    elif sf * sigma_c > allow:
        r_status = "AN2"  # centrifugal stress alone exceeds the allowable
    elif sf * (sigma_c + sigma_gb_rotor) > allow:
        r_status = "creep"
    else:
        r_status = "ok"
    T_allow, t_how = rc.allowable_temperature_K(sf * sigma_gb_ngv, life_h)
    data_limited = (
        t_how == "above_material_data"
    )  # T_allow is the top of the data, not a strength limit
    pf_max = (T_allow - tit_K) / (tit_K - T3_K) if (T_allow is not None and tit_K > T3_K) else None
    T_ngv = tit_K + pattern_factor * (tit_K - T3_K) if pattern_factor is not None else None
    fail = "above_material_data" if data_limited else "creep"
    if T_allow is None:
        n_status = "creep"  # stress too high even at the lowest tabulated temperature
    elif pf_max is not None and pf_max < 0:
        n_status = fail  # fails even with a perfectly uniform combustor exit (PF = 0)
    elif T_ngv is None:
        n_status = "unchecked_pattern_factor_missing"
    else:
        n_status = "ok" if T_ngv <= T_allow else fail
    for name, st in (("rotor", r_status), ("ngv", n_status)):
        if st in ("AN2", "creep"):
            active.append(f"{'AN2' if st == 'AN2' else 'creep_' + name}")
        elif st == "above_material_data":
            active.append(f"TIT_{name}_above_material_data")
    return CreepResult(material, life_h, sf, rotor_T_K, sigma_c, sigma_gb_rotor, allow, r_status, sigma_gb_ngv,
                       T_allow, pf_max, T_ngv, n_status, not active, active)  # fmt: skip

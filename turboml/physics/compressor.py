"""Centrifugal compressor meanline model (TurboML's own; primary compressor model).

What a meanline compressor model does
-------------------------------------
A centrifugal compressor sucks air in axially at the *inducer* (eye) (station 1), and the
spinning *impeller* flings it outward to its tip (station 2). That raises its pressure and its
kinetic energy. A *diffuser* (2 -> 3) then slows the air down and turns that kinetic energy
into more pressure.

A meanline model follows one representative streamline. At inlet and exit it builds the
**velocity triangles**: blade speed U = omega*r, absolute velocity C, relative velocity W = C - U.
The **Euler turbine equation** gives the work the blades put into each kg of air (no inlet swirl):

    dh_Euler = U2 * C_theta2

The flow "slips" behind the blades at the exit. With slip factor sigma and blade exit angle
beta2b (negative = backswept, measured from meridional):

    C_theta2 = sigma*U2 + C_m2*tan(beta2b),     sigma_Wiesner = 1 - sqrt(cos beta2b)/Z^0.7

Losses (Oh, Yoon & Chung 1997 "optimum set"):

* internal losses reduce the *useful* (isentropic) enthalpy rise: incidence (Conrad et al.
  1980), blade loading (Coppage et al. 1956), skin friction (Jansen 1967), tip clearance
  (Jansen 1967), wake mixing (Johnston & Dean 1966);
* parasitic losses add work that becomes heat: disk friction (Daily & Nece 1960),
  recirculation (Oh et al. 1997), leakage (Aungier 1995).

    h02 = h01 + dh_Euler + dh_parasitic            (actual total enthalpy rise)
    h02s = h01 + dh_Euler - dh_internal            (isentropic rise -> p02)

The correlations follow, line for line, TurboFlow's `centrifugal_compressor/loss_model_oh.py`
(Anderson et al., JOSS 2025, MIT licence), so the port can be verified against TurboFlow's
impeller (tests/test_compressor.py). TurboFlow's choices are kept, including its reading of
Conrad's incidence loss (whole inlet tangential relative velocity) and wake width 0.366.

The **vaneless diffuser** is integrated in radius as a 1-D annular flow with wall friction on
both walls (wall shear tau = Cf*rho*C^2/2; derivation in the docstring of `_diffuser`):

    d(r C_theta)/dr = -Cf * C * C_theta * r / (C_m * b)
    d(p0)/dr        = -Cf * rho * C^3 / (C_m * b)          (h0 constant: adiabatic)

Unlike TurboFlow's algebraic diffuser (lossless with the Oh set, docs/DECISIONS.md D-021), this
produces a real total-pressure loss. Cf is an input: no single universal value exists; the
default 0.005 is an ASSUMPTION within the commonly used 0.003-0.02 range (labelled in config).

Choke and surge (preliminary-design criteria)
---------------------------------------------
* ``choke_inducer``: no subsonic solution of continuity at the inducer (Mach -> 1).
* ``choke_throat``: mass flow exceeds the choked flow of the impeller throat, evaluated in the
  relative frame with relative stagnation conditions at the inducer mean radius;
  throat area = area_throat_ratio * inducer area (ratio is an assumption for the KJ-66).
* ``choke_impeller_exit`` / ``choke_diffuser``: no subsonic solution of continuity there.
* surge: the maximum-PR point of a speed line (zero slope). Points at lower flow are flagged
  ``surge_side``. Secondary flag ``diffuser_stall``: impeller-exit flow angle above the vaneless-
  diffuser critical angle of Senoo & Kinoshita (1977), as fitted in RadComp's
  ``radcompressor.diffuser.surge_critical_angle``.

All quantities SI; angles in degrees in the public interface.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from turboml.config import CONFIG_DIR
from turboml.gas import thermo

SLIP_MODELS = ("wiesner", "stodola")
F_INC = 0.5  # Conrad incidence factor (Oh 1997 range 0.5-0.7; TurboFlow value)
WAKE_WIDTH = 0.366  # Johnston & Dean mixing: wake fraction (TurboFlow default)


def air_viscosity(T: float) -> float:
    """Dynamic viscosity of air [Pa s], Sutherland's law (mu0 1.716e-5 at 273.15 K, S = 110.4 K;
    White, Viscous Fluid Flow, 3rd ed., Table 1-2)."""
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
@dataclass
class CentrifugalCompressorGeometry:
    """Impeller + vaneless diffuser geometry, SI units, angles in degrees.

    Blade angles from meridional; ``trailing_edge_angle_deg`` negative = backswept.
    """

    radius_hub_in_m: float
    radius_tip_in_m: float
    radius_out_m: float
    width_out_m: float
    number_of_blades: int
    tip_clearance_m: float
    leading_edge_angle_deg: float
    trailing_edge_angle_deg: float
    length_meridional_m: float
    length_axial_m: float
    area_throat_ratio: float
    diffuser_radius_out_m: float
    diffuser_width_out_m: float
    provenance: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(
        cls, path: str | Path = CONFIG_DIR / "components" / "kj66_compressor.yaml", **overrides
    ):
        raw = yaml.safe_load(Path(path).read_text())
        imp, vld = raw["impeller"], raw["vaneless_diffuser"]
        vals = {k: v["value"] for k, v in imp.items()}
        vals["diffuser_radius_out_m"] = vld["radius_out_m"]["value"]
        vals["diffuser_width_out_m"] = vld["width_out_m"]["value"]
        prov = {k: f"{v['kind']}: {v['source']}" for k, v in {**imp, **vld}.items()}
        vals.update(overrides)
        for k in overrides:
            prov[k] = "override"
        return cls(**vals, provenance=prov)

    # derived quantities
    @property
    def radius_mean_in_m(self) -> float:
        return 0.5 * (self.radius_hub_in_m + self.radius_tip_in_m)

    @property
    def area_in_m2(self) -> float:
        return math.pi * (self.radius_tip_in_m**2 - self.radius_hub_in_m**2)

    @property
    def area_out_m2(self) -> float:
        return 2.0 * math.pi * self.radius_out_m * self.width_out_m

    def tip_speed_m_s(self, n_rpm: float) -> float:
        return self.radius_out_m * n_rpm * math.pi / 30.0

    def to_turboflow(self) -> dict:
        return {
            "impeller": {
                "radius_hub_in": self.radius_hub_in_m, "radius_tip_in": self.radius_tip_in_m,
                "radius_out": self.radius_out_m, "width_out": self.width_out_m,
                "leading_edge_angle": self.leading_edge_angle_deg,
                "trailing_edge_angle": self.trailing_edge_angle_deg,
                "number_of_blades": self.number_of_blades, "length_axial": self.length_axial_m,
                "length_meridional": self.length_meridional_m, "tip_clearance": self.tip_clearance_m,
                "area_throat_ratio": self.area_throat_ratio,
            },
            "vaneless_diffuser": {
                "radius_in": self.radius_out_m, "width_in": self.width_out_m,
                "radius_out": self.diffuser_radius_out_m, "width_out": self.diffuser_width_out_m,
            },
        }  # fmt: skip


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass
class CompressorPoint:
    speed_rpm: float
    mass_flow_kg_s: float
    converged: bool
    failure_reason: str = ""
    pr_tt: float = np.nan  # stage total-to-total pressure ratio p03/p01
    eta_tt: float = np.nan  # stage isentropic total-to-total efficiency
    pr_tt_impeller: float = np.nan
    eta_tt_impeller: float = np.nan
    power_W: float = np.nan
    slip_factor: float = np.nan
    tip_speed_m_s: float = np.nan
    inlet_rel_mach_tip: float = np.nan
    exit_flow_angle_deg: float = np.nan  # alpha2 from meridional
    exit_mach: float = np.nan
    diffusion_ratio_w1t_w2: float = np.nan  # impeller relative-velocity diffusion (output only)
    diffuser_stall: bool = False
    T03_K: float = np.nan
    losses_J_kg: dict = field(default_factory=dict)


class _Fail(Exception):
    pass


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class CentrifugalCompressor:
    """Meanline centrifugal compressor: impeller (Oh 1997 loss set) + 1-D vaneless diffuser."""

    def __init__(
        self,
        geometry: CentrifugalCompressorGeometry,
        slip_model: str = "wiesner",
        delta_sigma: float = 0.0,
        diffuser_cf: float = 0.005,
        T0_in_K: float = 288.15,
        p0_in_Pa: float = 101325.0,
        diffuser_steps: int = 40,
        loss_convention: str = "oh_aungier",
        incidence_model: str = "blade_relative",
        loss_multiplier: float = 1.0,
    ):
        if slip_model not in SLIP_MODELS:
            raise ValueError(f"slip_model must be one of {SLIP_MODELS}")
        self.g, self.slip_model, self.delta_sigma = geometry, slip_model, delta_sigma
        self.diffuser_cf, self.T01, self.p01 = diffuser_cf, T0_in_K, p0_in_Pa
        self.n_steps = diffuser_steps
        if loss_convention not in ("oh_aungier", "turboflow"):
            raise ValueError("loss_convention must be 'oh_aungier' or 'turboflow'")
        # "oh_aungier" (default, classic): parasitic losses add work input; internal losses reduce
        #   the isentropic TOTAL enthalpy rise.  eta = (dh_E - internal) / (dh_E + parasitic).
        # "turboflow": rothalpy conserved (work = Euler only); all losses (incl. parasitic) are
        #   entropy, defined as STATIC enthalpy loss h2 - h(p2, s1). Used to verify the port.
        self.loss_convention = loss_convention
        if incidence_model not in ("blade_relative", "turboflow"):
            raise ValueError("incidence_model must be 'blade_relative' or 'turboflow'")
        # Conrad et al. (1980) incidence loss dh = f_inc * W_inc^2 / 2:
        # "blade_relative" (default): W_inc = W1 * sin(|beta1_flow| - beta1_blade) at the mean radius,
        #   i.e. the relative-velocity component lost by turning the flow onto the blade. Grows with
        #   off-design incidence; this is what produces a pressure-ratio peak (surge criterion).
        # "turboflow": W_inc = whole inlet tangential relative velocity (TurboFlow's reading).
        #   Independent of mass flow, so speed lines have no PR peak. Kept for the port test.
        self.incidence_model = incidence_model
        # Scales every impeller loss (verification: 0 -> loss-free impeller). Not a tuning knob.
        self.loss_multiplier = loss_multiplier
        self.air = thermo.air()
        self.h01 = self.air.h(self.T01)

    # slip ------------------------------------------------------------------------------
    def slip_factor(self) -> float:
        """sigma (radial-element definition) incl. the uncertainty offset delta_sigma.

        * wiesner: 1 - sqrt(cos b2b)/Z^0.7  (Wiesner 1967; limiting-radius correction not applied)
        * stodola: 1 - pi*cos(b2b)/Z        (Stodola 1927, as commonly quoted)
        """
        b2 = math.radians(self.g.trailing_edge_angle_deg)
        z = self.g.number_of_blades
        if self.slip_model == "wiesner":
            s = 1.0 - math.sqrt(math.cos(b2)) / z**0.7
        else:
            s = 1.0 - math.pi * math.cos(b2) / z
        return s + self.delta_sigma

    # station 1 -------------------------------------------------------------------------
    def _static_from_total(self, h0: float, T0: float, p0: float, v: float):
        T = self.air.T_from_h(h0 - 0.5 * v * v, T0)
        p = p0 * self.air.p_ratio_isentropic(T0, T)
        return T, p, p / (self.air.R * T)

    def _solve_continuity(self, mdot, area, h0, T0, p0, v_other=0.0, tag="inducer"):
        """Meridional velocity Cm on the subsonic branch of rho(Cm)*Cm*area = mdot.

        ``v_other`` = velocity component not in the area normal (swirl), held fixed.
        """

        def flux(cm):
            try:
                _, _, rho = self._static_from_total(h0, T0, p0, math.hypot(cm, v_other))
            except (ValueError, RuntimeError):
                return -1.0
            return rho * cm * area

        return _subsonic_root(flux, mdot, 1e-3, 1.2 * self.air.sound_speed(T0), f"choke_{tag}")

    # main evaluation -----------------------------------------------------------------
    def evaluate(self, mass_flow_kg_s: float, speed_rpm: float) -> CompressorPoint:
        """One operating point. Never raises for physical/solver failures (failures are data)."""
        pt = CompressorPoint(
            speed_rpm, mass_flow_kg_s, False, tip_speed_m_s=self.g.tip_speed_m_s(speed_rpm)
        )
        try:
            self._evaluate(pt)
            pt.converged = True
        except _Fail as f:
            pt.failure_reason = str(f)
        except (ValueError, RuntimeError, ZeroDivisionError, OverflowError) as err:
            pt.failure_reason = f"no_convergence: {type(err).__name__}: {err}"[:150]
        return pt

    def _evaluate(self, pt: CompressorPoint) -> None:
        g, air, mdot = self.g, self.air, pt.mass_flow_kg_s
        omega = pt.speed_rpm * math.pi / 30.0
        T01, p01, h01 = self.T01, self.p01, self.h01

        # ---- station 1: inducer inlet, axial, no swirl
        c1 = self._solve_continuity(mdot, g.area_in_m2, h01, T01, p01, tag="inducer")
        T1, p1, rho1 = self._static_from_total(h01, T01, p01, c1)
        a1 = air.sound_speed(T1)
        u1m, u1t = omega * g.radius_mean_in_m, omega * g.radius_tip_in_m
        w_t1 = -u1m  # relative tangential velocity at mean radius
        w1 = math.hypot(c1, w_t1)
        beta1 = math.degrees(math.atan2(w_t1, c1))
        pt.inlet_rel_mach_tip = math.hypot(c1, u1t) / a1
        mu1 = air_viscosity(T1)

        # ---- impeller throat choke (relative frame, mean radius)
        h0rel = air.h(T1) + 0.5 * w1 * w1
        T0rel = air.T_from_h(h0rel, T1)
        p0rel = p1 / air.p_ratio_isentropic(T0rel, T1)
        gam = air.gamma(T0rel)
        rho0rel = p0rel / (air.R * T0rel)
        a0rel = air.sound_speed(T0rel)
        m_choke = (
            g.area_throat_ratio
            * g.area_in_m2
            * rho0rel
            * a0rel
            * (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1)))
        )
        if mdot > m_choke:
            raise _Fail("choke_throat")

        # ---- station 2: impeller exit
        u2 = omega * g.radius_out_m
        sigma = self.slip_factor()
        pt.slip_factor = sigma
        tb2 = math.tan(math.radians(g.trailing_edge_angle_deg))

        def exit_state(cm2, rho2_guess):
            ct2 = sigma * u2 + cm2 * tb2
            dh_euler = u2 * ct2
            wt2 = ct2 - u2
            w2, c2 = math.hypot(cm2, wt2), math.hypot(cm2, ct2)
            alpha2 = math.degrees(math.atan2(ct2, cm2))
            beta2 = math.degrees(math.atan2(wt2, cm2))
            rho2, T2 = rho2_guess, T01
            for _ in range(30):
                L = self._losses(c1, w1, w_t1, beta1, rho1, mu1, cm2, ct2, c2, w2, alpha2, beta2,
                                 rho2, u2, dh_euler, mdot, T2=T2)  # fmt: skip
                if self.loss_convention == "oh_aungier":
                    h02 = h01 + dh_euler + L["parasitic"]
                    T02 = air.T_from_h(h02, T01)
                    T02s = air.T_from_h(h01 + dh_euler - L["internal"], T01)
                    p02 = p01 * air.p_ratio_isentropic(T01, T02s)
                    T2, p2, rho2_new = self._static_from_total(h02, T02, p02, c2)
                else:
                    h02 = h01 + dh_euler
                    T02 = air.T_from_h(h02, T01)
                    h2 = h02 - 0.5 * c2 * c2
                    T2 = air.T_from_h(h2, T01)
                    T2s = air.T_from_h(h2 - L["internal"] - L["parasitic"], T01)
                    p2 = p01 * air.p_ratio_isentropic(T01, T2s)
                    p02 = p2 / air.p_ratio_isentropic(T02, T2)
                    rho2_new = p2 / (air.R * T2)
                if abs(rho2_new - rho2) < 1e-11 * rho2:
                    break
                rho2 = rho2_new
            return dict(ct2=ct2, c2=c2, w2=w2, alpha2=alpha2, h02=h02, T02=T02, p02=p02, T2=T2,
                        p2=p2, rho2=rho2_new, dh_euler=dh_euler, L=L)  # fmt: skip

        def flux2(cm2):
            st = exit_state(cm2, rho1)
            return st["rho2"] * cm2 * g.area_out_m2, st

        def flux2_safe(cm2):
            try:
                return flux2(cm2)[0]
            except (ValueError, RuntimeError, OverflowError):
                return -1.0

        cm2 = _subsonic_root(
            flux2_safe, mdot, 1.0, 1.5 * air.sound_speed(T01), "choke_impeller_exit"
        )
        st = exit_state(cm2, rho1)

        pt.exit_flow_angle_deg = st["alpha2"]
        pt.diffusion_ratio_w1t_w2 = math.hypot(c1, u1t) / st["w2"]
        pt.exit_mach = st["c2"] / air.sound_speed(st["T2"])
        pt.losses_J_kg = {k: v for k, v in st["L"].items() if k not in ("internal", "parasitic")}
        T02s_imp = air.T_isentropic(T01, st["p02"] / p01)
        pt.pr_tt_impeller = st["p02"] / p01
        pt.eta_tt_impeller = (air.h(T02s_imp) - h01) / (st["h02"] - h01)
        pt.power_W = mdot * (st["h02"] - h01)
        pt.T03_K = st["T02"]

        # ---- vaneless diffuser 2 -> 3
        p03, dp0 = self._diffuser(st, cm2, mdot)
        pt.losses_J_kg["vaneless_diffuser_dp0_Pa"] = dp0
        T03s = air.T_isentropic(T01, p03 / p01)
        pt.pr_tt = p03 / p01
        pt.eta_tt = (air.h(T03s) - h01) / (st["h02"] - h01)
        pt.diffuser_stall = bool(st["alpha2"] > _senoo_critical_angle(g, pt.exit_mach))

    # losses ---------------------------------------------------------------------------
    def _losses(self, c1, w1, w_t1, beta1, rho1, mu1, cm2, ct2, c2, w2, alpha2, beta2, rho2, u2,
                dh_euler, mdot, T2) -> dict:  # fmt: skip
        """Oh et al. (1997) loss set [J/kg], equations as in TurboFlow loss_model_oh.py."""
        g = self.g
        z, r1t, r1h, r2, b2 = (
            g.number_of_blades,
            g.radius_tip_in_m,
            g.radius_hub_in_m,
            g.radius_out_m,
            g.width_out_m,
        )
        cos_b1, cos_b2 = math.cos(math.radians(beta1)), math.cos(math.radians(beta2))
        # incidence (Conrad et al. 1980), see incidence_model in __init__
        if self.incidence_model == "turboflow":
            inc = F_INC * 0.5 * w_t1**2
        else:
            w_inc = w1 * math.sin(math.radians(abs(beta1) - abs(g.leading_edge_angle_deg)))
            inc = F_INC * 0.5 * w_inc**2
        # blade loading (Coppage et al. 1956); TurboFlow uses actual work h0_out - h0_in, here the
        # Euler work (difference = parasitic work, second order) to avoid an algebraic loop
        df = (
            1
            - w2 / w1
            + 0.75 * dh_euler * w2 / ((z / math.pi * (1 - r1t / r2) + 2 * r1t / r2) * w1 * u2**2)
        )
        bld = 0.05 * df**2 * u2**2
        # skin friction (Jansen 1967)
        L_b = (
            math.pi
            / 8
            * (2 * r2 - (r1t + r1h) - b2 + 2 * g.length_axial_m)
            * (2 / (cos_b1 + cos_b2))
        )
        D_h = 2 * r2 * cos_b2 / (z / math.pi + 2 * r2 * cos_b2 / b2) + 0.5 * (r1t + r1h) / r2 * cos_b1 / (
            z / math.pi + (r1t + r1h) / (r1t - r1h) * cos_b1)  # fmt: skip
        w_avg = (c1 + c2 + 3 * w1 + 3 * w2) / 8
        Re = u2 * D_h * rho1 / mu1
        cf = 0.0412 * Re**-0.1925
        sf = 2 * cf * L_b / D_h * w_avg**2
        # tip clearance (Jansen 1967)
        cl = 0.6 * g.tip_clearance_m / b2 * ct2 * math.sqrt(
            4 * math.pi / (b2 * z) * (r1t**2 - r1h**2) / ((r2 - r1t) * (1 + rho2 / rho1)) * ct2 * c1)  # fmt: skip
        # wake mixing (Johnston & Dean 1966), diffuser width = impeller exit width (b* = 1)
        mix = (
            1
            / (1 + math.tan(math.radians(alpha2)) ** 2)
            * ((1 - WAKE_WIDTH - 1.0) / (1 - WAKE_WIDTH)) ** 2
            * 0.5
            * c2**2
        )
        # parasitic: recirculation (Oh et al. 1997)
        rc = 8e-5 * math.sinh(3.5 * math.radians(alpha2) ** 3) * df**2 * u2**2
        # disk friction (Daily & Nece 1960)
        mu2 = air_viscosity(T2)
        Re_df = rho2 * u2 * r2 / mu2
        f_df = 2.67 / Re_df**0.5 if Re_df < 3e5 else 0.0622 / Re_df**0.2
        dfr = f_df * 0.5 * (rho1 + rho2) * r2**2 * u2**3 / (4 * mdot)
        # leakage (Aungier 1995)
        r_avg = 0.5 * (g.radius_mean_in_m + r2)
        b_avg = 0.5 * ((r1t - r1h) + b2)
        dp = mdot * (r2 * ct2) / (z * r_avg * b_avg * g.length_meridional_m)
        u_cl = 0.816 * math.sqrt(2 * max(dp, 0.0) / rho2)
        m_cl = rho2 * z * g.tip_clearance_m * g.length_meridional_m * u_cl
        lk = m_cl * u_cl * u2 / (2 * mdot)
        k = self.loss_multiplier
        inc, bld, sf, cl, mix, rc, dfr, lk = (k * v for v in (inc, bld, sf, cl, mix, rc, dfr, lk))
        internal = inc + bld + sf + cl + mix
        parasitic = rc + dfr + lk
        return {"incidence": inc, "blade_loading": bld, "skin_friction": sf, "tip_clearance": cl,
                "wake_mixing": mix, "recirculation": rc, "disk_friction": dfr, "leakage": lk,
                "internal": internal, "parasitic": parasitic, "diffusion_factor": df}  # fmt: skip

    # vaneless diffuser ----------------------------------------------------------------
    def _diffuser(self, st: dict, cm2: float, mdot: float) -> tuple[float, float]:
        """1-D vaneless diffuser with wall friction; returns (p03, total-pressure loss).

        Derivation: wall shear tau = Cf rho C^2/2 acts on both walls (area 2 * 2 pi r dr) against
        the velocity. Torque balance on the annulus gives d(mdot r C_theta) = -tau (C_theta/C) r
        * 4 pi r dr; with mdot = rho C_m 2 pi r b:  d(r C_theta)/dr = -Cf C C_theta r / (C_m b).
        Dissipated power tau*C*area divided by the volume flow mdot/rho gives the total-pressure
        loss:  dp0/dr = -Cf rho C^3 / (C_m b).  Adiabatic: h0 = const.
        """
        g = self.g
        r2, r3 = g.radius_out_m, g.diffuser_radius_out_m
        b2, b3 = g.width_out_m, g.diffuser_width_out_m
        h0, T0 = st["h02"], st["T02"]
        cf = self.diffuser_cf

        def width(r):
            return b2 + (b3 - b2) * (r - r2) / (r3 - r2)

        def deriv(r, y):
            rct, p0 = y
            ct = rct / r
            b = width(r)
            cm = self._solve_continuity(
                mdot, 2 * math.pi * r * b, h0, T0, p0, v_other=ct, tag="diffuser"
            )
            c = math.hypot(cm, ct)
            _, _, rho = self._static_from_total(h0, T0, p0, c)
            return np.array([-cf * c * ct * r / (cm * b), -cf * rho * c**3 / (cm * b)])

        y = np.array([r2 * st["ct2"], st["p02"]])
        rs = np.linspace(r2, r3, self.n_steps + 1)
        for i in range(self.n_steps):  # classical RK4
            r, h = rs[i], rs[i + 1] - rs[i]
            k1 = deriv(r, y)
            k2 = deriv(r + h / 2, y + h / 2 * k1)
            k3 = deriv(r + h / 2, y + h / 2 * k2)
            k4 = deriv(r + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return float(y[1]), float(st["p02"] - y[1])

    # speed lines ---------------------------------------------------------------------
    def speed_line(self, speed_rpm: float, mass_flows) -> pd.DataFrame:
        """Evaluate a speed line and locate the (preliminary-design) surge point.

        Surge point = highest mass flow at which either criterion trips:
          * ``peak_pr``: an INTERIOR maximum of PR along the line (zero slope); a maximum at the
            lowest evaluated flow is not a peak and is ignored;
          * ``diffuser_stall``: Senoo & Kinoshita critical flow angle exceeded.
        Points at or below the surge flow get ``surge_side = True``. Choke: highest converged flow
        (the next point failed with a ``choke_*`` reason). Failures stay in the table.
        """
        rows = [asdict(self.evaluate(m, speed_rpm)) for m in np.sort(np.asarray(mass_flows, float))]
        df = pd.DataFrame(rows).drop(columns="losses_J_kg")
        ok = df.converged
        df["is_peak_pr"] = False
        df["surge_side"] = False
        df["surge_criterion"] = ""
        if ok.any():
            okd = df[ok]
            ipk = okd.pr_tt.idxmax()
            m_peak = okd.loc[ipk, "mass_flow_kg_s"] if ipk != okd.index[0] else -np.inf
            df.loc[ipk, "is_peak_pr"] = ipk != okd.index[0]
            stalled = okd[okd.diffuser_stall]
            m_stall = stalled.mass_flow_kg_s.max() if len(stalled) else -np.inf
            m_surge = max(m_peak, m_stall)
            if np.isfinite(m_surge):
                df.loc[ok & (df.mass_flow_kg_s <= m_surge), "surge_side"] = True
                isurge = (df.mass_flow_kg_s - m_surge).abs().idxmin()
                df.loc[isurge, "surge_criterion"] = (
                    "peak_pr" if m_peak >= m_stall else "diffuser_stall"
                )
        df["slip_model"], df["delta_sigma"], df["diffuser_cf"] = (
            self.slip_model,
            self.delta_sigma,
            self.diffuser_cf,
        )
        return df


def _subsonic_root(flux, target: float, lo: float, hi: float, choke_label: str) -> float:
    """Root of flux(c) = target on the subsonic branch (below the maximum of flux on [lo, hi]).

    flux(c) = rho(c)*c*A rises with c up to the local sonic condition, then falls. The maximum
    is located by bounded scalar optimisation; if it is below ``target`` the passage is choked.
    """
    from scipy.optimize import minimize_scalar

    opt = minimize_scalar(
        lambda c: -flux(c), bounds=(lo, hi), method="bounded", options={"xatol": 1e-6}
    )
    c_star, f_star = opt.x, -opt.fun
    if f_star < target:
        raise _Fail(choke_label)
    if flux(lo) > target:
        raise _Fail("no_convergence: flux above target at lower bracket")
    return brentq(lambda c: flux(c) - target, lo, c_star, xtol=1e-10)


def _senoo_critical_angle(g: CentrifugalCompressorGeometry, mach2: float) -> float:
    """Vaneless-diffuser critical inlet flow angle [deg from meridional] (Senoo & Kinoshita 1977),
    via RadComp's fit ``radcompressor.diffuser.surge_critical_angle(r5, r4, b4, m2)``."""
    from radcompressor.diffuser import surge_critical_angle

    try:
        return float(
            surge_critical_angle(g.diffuser_radius_out_m, g.radius_out_m, g.width_out_m, mach2)
        )
    except Exception:  # noqa: BLE001 - outside fit range -> no flag
        return 90.0


def surge_margin(m_operating: float, m_surge: float) -> float:
    """Surge margin at constant speed, SM = (m_op - m_surge) / m_op (mass-flow definition)."""
    return (m_operating - m_surge) / m_operating

"""Fast single-stage axial turbine meanline model (our own; TurboFlow's formulation, marching solver).

Why a second turbine model
--------------------------
The TurboFlow wrapper (`turboml.physics.turbine.AxialTurbine`) takes ~3 s per operating point,
because it solves all flow equations at once with a general-purpose root finder and CoolProp
properties. Engine matching (Phase 3) and dataset generation (Phase 4) need thousands of turbine
evaluations per design, so this module re-implements the same physics as a *marching* solver:
the flow is computed plane by plane (stator inlet -> stator throat -> stator exit -> rotor inlet
-> rotor throat -> rotor exit), and only one scalar unknown is iterated at a time. Properties come
from our NASA-polynomial gas model (`turboml.gas.thermo`), so combustion products are supported.
TurboFlow stays available as the higher-fidelity cross-check (and as a possible multi-fidelity level).

Physics (per blade row, along the mean-radius streamline)
---------------------------------------------------------
* Velocity triangles: absolute velocity C, blade speed U = omega r, relative velocity W = C - U.
  Angles from the axial direction, degrees; TurboFlow sign convention (stator exit angle > 0,
  rotor exit relative angle < 0).
* Energy: rothalpy I = h + W^2/2 - U^2/2 is constant through a row (no heat transfer); in the stator
  (U = 0) this is just h0 = const.
* Continuity: m = rho W cos(beta) A (1 - blockage) at inlet and exit; m = rho W A_throat (1 - b) at
  the throat (the throat area is normal to the flow).
* Losses (Benner et al. 1997/2006 set as implemented in TurboFlow 0.1.18, `loss_model_benner.py`,
  MIT licence, ported with attribution): profile, trailing-edge, secondary, tip-clearance and
  incidence losses, combined as a stagnation-pressure loss coefficient
      Y = (p0rel_is - p0rel_out) / (p0rel_out - p_out),
  where p0rel_is is the relative stagnation pressure an isentropic row would reach. Y fixes the
  exit entropy: s_out = s_in + R ln(p0rel_is / p0rel_out).
* Exit flow angle (subsonic): Aungier deviation model. Deviation shrinks from its low-speed value at
  Ma = 0.5 to zero at the critical Mach number.
* Choking ("critical Mach number" criterion, TurboFlow): a row chokes when its throat reaches the
  critical Mach number of a nozzle with the same efficiency eta (Ma_crit < 1 when eta < 1). Above
  that, the mass flow through the row is fixed and the exit flow angle is set by continuity
  (supersonic expansion past the throat), while the exit pressure keeps falling.
* Inter-row gap: angular momentum and stagnation enthalpy conserved, density assumed constant
  (TurboFlow's interspace model).

Operating regimes (how the marching solver handles choking)
-----------------------------------------------------------
A. Neither row choked: unknown = mass flow; solve p_exit(m) = target.
B. Stator choked, rotor not: m = stator critical flow; unknown = stator exit static pressure p1.
C. Both choked: m = stator critical flow; p1 such that the rotor is exactly critical; rotor exit at
   the target pressure.
D. Rotor choked first: m = rotor critical flow (stator subsonic); rotor exit at the target pressure.
Failures are returned as data (`failure_reason`): "limit_loading" (exit cannot expand further,
cos(beta) > 1), "no_convergence", "invalid_input".

Differences from TurboFlow (all switchable for the port verification)
---------------------------------------------------------------------
* `incidence_sign`: TurboFlow 0.1.18 computes the Benner incidence parameter with
  ``cascade_type = ["cascade_type"]`` (a list literal instead of the geometry entry), so its intended
  rotor sign (+1) is never used and every row gets -1. Default here: "turboflow" (-1 for every row),
  because it reproduces TurboFlow AND fits the Kofskey measurements better: at 50 % speed (+27 deg
  rotor incidence) it predicts eta_ts within 0.1 pts of the data, the intended sign ("rotor_plus")
  2.6 pts low (validation/phase3/kofskey_fast_summary.yaml). Which sign Benner et al. (1997) actually
  define for rotor angles is OPEN until the paper is checked (docs/PHASE3_REPORT.md).
* Gas properties: NASA polynomials (ideal gas, any fixed composition) instead of CoolProp. Below
  200 K (only reached in cold-air rig tests) h and s0 are extrapolated with constant cp.
* Viscosity: Sutherland's law for air (also used for combustion products: ASSUMPTION, a few-% error
  in Re, i.e. ~1 % in the profile loss through Re^-0.4).

Preliminary-design model only; validated range in docs/VALIDATION.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol

import yaml

from turboml.gas.thermo import GasMixture, air

P_REF = 101325.0  # Pa, entropy reference of the NASA polynomials
_D2R = math.pi / 180.0


def _sind(x):
    return math.sin(x * _D2R)


def _cosd(x):
    return math.cos(x * _D2R)


def _tand(x):
    return math.tan(x * _D2R)


def _atand(x):
    return math.atan(x) / _D2R


def _acosd(x):
    return math.acos(x) / _D2R


def _asind(x):
    return math.asin(x) / _D2R


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def air_viscosity(T: float) -> float:
    """Sutherland's law for air [Pa s] (White, Viscous Fluid Flow, Table 1-2)."""
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


class _ModelLimit(Exception):
    """A physical limit of the flow model was reached (handled by the solver, reported as data)."""


class _Choke(_ModelLimit):
    """Continuity has no subsonic solution in this plane (the row is choked), or an inner iteration failed."""


class _LimitLoading(_ModelLimit):
    """A choked row's exit can't expand any further (cos(beta) would exceed 1)."""


# ======================================================================================
# Gas: fast inline evaluation of the GasMixture NASA polynomials (real cp(T), ideal gas)
# ======================================================================================
class _Gas:
    """Same polynomials as `GasMixture`, evaluated inline (no per-call dispatch) for speed."""

    T_LOW, T_BREAK, T_HIGH = 200.0, 1000.0, 6000.0  # below 200 K: constant-cp extrapolation

    def __init__(self, gas: GasMixture):
        self.g, self.R = gas, gas.R
        self._cpc = [tuple(a[:7]) for a in gas._a]
        self._hc = [
            (a[0], a[1], a[2], a[3] / 2, a[4] / 3, a[5] / 4, a[6] / 5, a[7]) for a in gas._a
        ]
        self._sc = [
            (a[0] / 2, a[1], a[2], a[3], a[4] / 2, a[5] / 3, a[6] / 4, a[8]) for a in gas._a
        ]
        self._cp200 = gas.cp(200.0)
        self._h200 = gas.h(200.0)
        self._s200 = gas.s0(200.0)

    def _idx(self, T):
        if T <= self.T_BREAK:
            return 0
        if T > self.T_HIGH:
            raise ValueError(f"T = {T:.0f} K above the NASA polynomial range")
        return 1

    def cp(self, T):
        if T < self.T_LOW:
            return self._cp200
        a = self._cpc[self._idx(T)]
        return a[0] / (T * T) + a[1] / T + a[2] + T * (a[3] + T * (a[4] + T * (a[5] + T * a[6])))

    def h(self, T):
        if T < self.T_LOW:
            return self._h200 + self._cp200 * (T - self.T_LOW)
        a = self._hc[self._idx(T)]
        return (
            -a[0] / T
            + a[1] * math.log(T)
            + T * (a[2] + T * (a[3] + T * (a[4] + T * (a[5] + T * a[6]))))
            + a[7]
        )

    def s0(self, T):
        if T < self.T_LOW:
            return self._s200 + self._cp200 * math.log(T / self.T_LOW)
        a = self._sc[self._idx(T)]
        return (
            -a[0] / (T * T)
            - a[1] / T
            + a[2] * math.log(T)
            + T * (a[3] + T * (a[4] + T * (a[5] + T * a[6])))
            + a[7]
        )

    def gamma(self, T):
        cp = self.cp(T)
        return cp / (cp - self.R)

    def T_h(self, h, T):
        for _ in range(50):
            dT = (self.h(T) - h) / self.cp(T)
            T -= dT
            if T <= 1.0:
                raise _Choke("T(h) below 1 K")
            if abs(dT) < 1e-11 * T:
                return T
        raise _Choke("T(h) did not converge")

    def T_ps(self, p, s, T):
        """Temperature with pressure p and entropy s."""
        target = s + self.R * math.log(p / P_REF)
        for _ in range(50):
            dT = (self.s0(T) - target) * T / self.cp(T)
            T -= dT
            if T <= 1.0:
                raise _Choke("T(p, s) below 1 K")
            if abs(dT) < 1e-11 * T:
                return T
        raise _Choke("T(p, s) did not converge")

    def p_Ts(self, T, s):
        return P_REF * math.exp((self.s0(T) - s) / self.R)

    def s_Tp(self, T, p):
        return self.s0(T) - self.R * math.log(p / P_REF)


# ======================================================================================
# Geometry
# ======================================================================================
_GEOM_KEYS = (
    "radius_hub_in", "radius_hub_out", "radius_tip_in", "radius_tip_out", "pitch", "chord",
    "stagger_angle", "opening", "leading_edge_angle", "leading_edge_wedge_angle",
    "leading_edge_diameter", "trailing_edge_thickness", "maximum_thickness", "tip_clearance",
    "throat_location_fraction",
)  # fmt: skip


@dataclass
class AxialStageGeometry:
    """Single axial stage [stator, rotor] in TurboFlow's geometry format (SI units, angles in degrees).

    Each field is a 2-list (stator, rotor). Angles: TurboFlow convention (from axial; stator exit
    and stagger positive, rotor stagger negative, rotor leading-edge angle positive).
    """

    radius_hub_in: list[float]
    radius_hub_out: list[float]
    radius_tip_in: list[float]
    radius_tip_out: list[float]
    pitch: list[float]
    chord: list[float]
    stagger_angle: list[float]
    opening: list[float]
    leading_edge_angle: list[float]
    leading_edge_wedge_angle: list[float]
    leading_edge_diameter: list[float]
    trailing_edge_thickness: list[float]
    maximum_thickness: list[float]
    tip_clearance: list[float]
    throat_location_fraction: list[float]
    cascade_type: list[str] = field(default_factory=lambda: ["stator", "rotor"])

    @classmethod
    def from_turboflow(cls, geom: dict) -> AxialStageGeometry:
        def val(v):
            return [
                float(eval(x, {"np": __import__("numpy")})) if isinstance(x, str) else float(x)
                for x in v
            ]  # noqa: S307

        return cls(
            **{k: val(geom[k]) for k in _GEOM_KEYS},
            cascade_type=list(geom.get("cascade_type", ["stator", "rotor"])),
        )

    @classmethod
    def from_turboflow_yaml(cls, path: str | Path) -> AxialStageGeometry:
        return cls.from_turboflow(
            yaml.safe_load(Path(path).read_text(encoding="utf-8"))["geometry"]
        )

    def to_turboflow(self) -> dict:
        return {
            "cascade_type": list(self.cascade_type),
            **{k: list(getattr(self, k)) for k in _GEOM_KEYS},
        }

    def cascades(self) -> list[_Cascade]:  # noqa: F821 (defined below)
        return [_Cascade(self, i) for i in range(2)]

    @property
    def rotor_tip_radius_m(self) -> float:
        return max(self.radius_tip_in[1], self.radius_tip_out[1])

    @property
    def rotor_annulus_area_m2(self) -> float:
        """Rotor exit annulus area (used for the AN^2 stress proxy)."""
        return math.pi * (self.radius_tip_out[1] ** 2 - self.radius_hub_out[1] ** 2)

    def copy_with(self, **changes) -> AxialStageGeometry:
        d = {f.name: list(getattr(self, f.name)) for f in fields(self)}
        d.update(changes)
        return AxialStageGeometry(**d)


class _Cascade:
    """Derived geometry of one blade row (port of TurboFlow `calculate_full_geometry`)."""

    def __init__(self, g: AxialStageGeometry, i: int):
        self.type = g.cascade_type[i]
        self.is_rotor = self.type == "rotor"
        fr = g.throat_location_fraction[i]
        rh_in, rh_out, rt_in, rt_out = (
            g.radius_hub_in[i],
            g.radius_hub_out[i],
            g.radius_tip_in[i],
            g.radius_tip_out[i],
        )
        rh_th, rt_th = (1 - fr) * rh_in + fr * rh_out, (1 - fr) * rt_in + fr * rt_out
        self.r_in, self.r_out = (rt_in + rh_in) / 2, (rt_out + rh_out) / 2
        self.r_throat = (1 - fr) * self.r_in + fr * self.r_out
        self.hub_tip_in = rh_in / rt_in
        h_in, h_out, h_th = rt_in - rh_in, rt_out - rh_out, rt_th - rh_th
        self.height = (h_in + h_out) / 2
        self.A_in = math.pi * (rt_in**2 - rh_in**2)
        self.A_out = math.pi * (rt_out**2 - rh_out**2)
        self.pitch, self.chord, self.opening = g.pitch[i], g.chord[i], g.opening[i]
        self.A_throat = 2 * math.pi * self.r_throat * h_th * self.opening / self.pitch
        if not 0.0 < self.A_throat < self.A_out:
            raise ValueError(f"{self.type}: throat area must be between 0 and the exit area")
        self.gauging = _acosd(self.A_throat / self.A_out)  # unsigned, degrees from axial
        self.sign = -1.0 if i % 2 else 1.0  # TurboFlow: stator exit angles > 0, rotor < 0
        self.stagger = g.stagger_angle[i]
        self.axial_chord = self.chord * _cosd(self.stagger)
        self.theta_in = g.leading_edge_angle[i]
        self.wedge = g.leading_edge_wedge_angle[i]
        self.le_diameter = g.leading_edge_diameter[i]
        self.t_te = g.trailing_edge_thickness[i]
        self.t_max = g.maximum_thickness[i]
        self.tip_clearance = g.tip_clearance[i]
        # Aungier low-speed deviation (depends on geometry only)
        beta_g = 90.0 - self.gauging
        cg = _cosd(self.gauging)
        self.delta0 = _asind(min(1.0, cg * (1 + (1 - cg) * (beta_g / 90.0) ** 2))) - beta_g
        # geometry-only parts of the Benner loss correlations (precomputed for speed)
        self.index = i
        self.r_sc = self.pitch / self.chord
        self.f_hub = _interp(max(self.hub_tip_in, 0.5), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                             (2.15, 1.7, 1.35, 1.12, 1.0, 1.0) if self.is_rotor else (1.4, 1.18, 1.05, 1.0, 1.0, 1.0))  # fmt: skip
        self.tmax_ratio = (self.t_max / self.chord) / 0.2
        r_to = min(0.4, self.t_te / self.opening)
        self.te_r = _interp(r_to, (0, 0.2, 0.4), (0, 0.045, 0.15))
        self.te_i = _interp(r_to, (0, 0.2, 0.4), (0, 0.025, 0.075))
        self.AR = self.height / self.chord
        self.cs = _cosd(self.stagger)
        self.BSx = self.axial_chord / self.pitch
        self.chi_geom = ((self.le_diameter / self.pitch) ** (-0.05) * self.wedge ** (-0.2)
                         * (_cosd(self.theta_in) / _cosd(self.gauging)) ** (-1.4))  # fmt: skip
        self.cl_factor = (0.37 * self.chord / self.height * (self.tip_clearance / self.height) ** 0.78
                          if self.is_rotor else 0.0)  # fmt: skip


# ======================================================================================
# Benner loss model (port of TurboFlow 0.1.18 loss_model_benner.py; equations unchanged)
# ======================================================================================
_INC_C1 = (
    -6.149e-5,
    +1.327e-3,
    -2.506e-4,
    -1.542e-4,
    +9.017e-5,
    +1.106e-5,
    -5.318e-6,
    +3.711e-7,
)  # x^1..x^8
_INC_C2 = (-8.72e-4, 1.358e-4)  # x^1, x^2
_CHI_EX = 5.0
_INC_EX_VALUE = sum(a * _CHI_EX ** (i + 1) for i, a in enumerate(_INC_C1))
_INC_EX_SLOPE = sum((i + 1) * a * _CHI_EX**i for i, a in enumerate(_INC_C1))


def _inc_increment(chi, limit=0.5):
    """Kinetic-energy loss increment due to incidence (Benner 1997 polynomial; linear beyond chi = 5;
    TurboFlow's log-sum-exp smooth minimum with the 0.5 cap, alpha = 25)."""
    if chi > _CHI_EX:
        d = _INC_EX_VALUE + _INC_EX_SLOPE * (chi - _CHI_EX)
    elif chi >= 0:
        x = chi
        c = _INC_C1
        d = x * (
            c[0]
            + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * c[7]))))))
        )
    else:
        d = chi * (_INC_C2[0] + chi * _INC_C2[1])
    a, m = 25.0, min(d, limit)
    return m - math.log(math.exp(-a * (d - m)) + math.exp(-a * (limit - m))) / a


def _nozzle_blades(r_sc, angle_out):
    phi = 90 - angle_out
    r_sc_min = (0.46 + phi / 77) if phi < 30 else (0.614 + phi / 130)
    X = r_sc - r_sc_min
    A = (0.025 + (27 - phi) / 530) if phi < 27 else (0.025 + (27 - phi) / 3085)
    B = 0.1583 - phi / 1640
    if phi < 30:
        C = 0.08 * ((phi / 30) ** 2 - 1)
        return A + B * X**2 + C * X**3
    return A + B * abs(X) ** (1 + phi / 30)


def _impulse_blades(r_sc, angle_out):
    phi = 90 - angle_out
    r_sc_min = 0.224 + 1.575 * (phi / 90) - (phi / 90) ** 2
    X = r_sc - r_sc_min
    A = 0.242 - phi / 151 + (phi / 127) ** 2
    B = (0.3 + (30 - phi) / 50) if phi < 30 else (0.3 + (30 - phi) / 275)
    C = 0.88 - phi / 42.4 + (phi / 72.8) ** 2
    return A + B * X**2 - C * X**3


def _benner(
    c, Re_in, Re, Ma_in, Ma_out, p0is, p_in, p0rel_out, p_out, b_in, b_out, g, inc_sign, dh_ratio
):
    """Benner et al. loss coefficients of one row; returns (profile, trailing, secondary, clearance,
    incidence). Inputs as TurboFlow passes them (Re, relative Mach, pressures, relative angles [deg]).
    """
    Re_in, Re = max(1e-3, Re_in), max(1e-3, Re)
    Ma_in, Ma_out = max(1e-3, Ma_in), max(1e-3, Ma_out)
    theta_in = c.theta_in
    dh = dh_ratio * (Re_in / 3e5) ** (-1 / 7)

    # profile loss (Kacker-Okapuu form with Benner's modifications)
    f_Re = (Re / 2e5) ** (-0.4) if Re < 2e5 else (1.0 if Re <= 1e6 else (Re / 1e6) ** (-0.2))
    f_Ma = 1 + 60 * (Ma_out - 1) ** 2 if Ma_out > 1 else 1.0
    a = c.f_hub * Ma_in - 0.4
    Y_shock = (
        max(0.0, 0.75 * a**1.75 * c.hub_tip_in * (p0is - p_in) / (p0rel_out - p_out))
        if a > 0
        else 0.0
    )
    K1 = 1.0 if Ma_out < 0.2 else ((1 - 1.25 * (Ma_out - 0.2)) if Ma_out < 1.0 else 0.0)
    Kp = max(0.1, 1 - (Ma_in / Ma_out) ** 2 * (1 - K1))
    ang = max(abs(b_out), 40.0)
    Yr, Yi = _nozzle_blades(c.r_sc, ang), _impulse_blades(c.r_sc, ang)
    ratio = theta_in / b_out
    Y_p = max(Yr - abs(ratio) * ratio * (Yi - Yr), 0.8 * Yr)
    if ratio < 0:
        Y_p *= c.tmax_ratio ** (-ratio)
    Y_p = f_Re * f_Ma * 0.914 * (2 / 3 * Y_p * Kp + Y_shock)

    # trailing-edge loss
    d_phi2 = max(c.te_r - abs(ratio) * ratio * (c.te_i - c.te_r), c.te_i / 2)
    Y_te = 1 / (1 - d_phi2) - 1

    # secondary loss
    cb_in, cb_out = _cosd(b_in), _cosd(b_out)
    CR = cb_in / cb_out
    if c.AR <= 2:
        Y_s = (0.038 + 0.41 * math.tanh(1.2 * dh)) / (
            math.sqrt(c.cs) * CR * c.AR**0.55 * (cb_out / c.cs) ** 0.55
        )
    else:
        Y_s = (0.052 + 0.56 * math.tanh(1.2 * dh)) / (
            math.sqrt(c.cs) * CR * c.AR * (cb_out / c.cs) ** 0.55
        )

    # tip-clearance loss
    t_in, t_out = _tand(b_in), _tand(b_out)
    tm = (t_in + t_out) / 2
    cos_am2 = 1.0 / (1.0 + tm * tm)  # cos^2 of the mean angle atan(tm)
    Z = 4 * (t_in - t_out) ** 2 * cb_out**2 / math.sqrt(cos_am2)
    Y_cl = c.cl_factor * Z

    # incidence loss
    chi = c.chi_geom * inc_sign * (b_in - theta_in)
    dPhi = _inc_increment(chi)
    e = g / (g - 1)
    Y_inc = ((1 - (g - 1) / 2 * Ma_out**2 * (1 / (1 - dPhi) - 1)) ** (-e) - 1) / (
        1 - (1 + (g - 1) / 2 * Ma_out**2) ** (-e)
    )

    # penetration depth of the secondary flows
    F_t = 2 / c.BSx * cos_am2 * (abs(t_in) + abs(t_out))
    Z_TE = min(0.10 * F_t**0.79 / math.sqrt(CR) / c.AR**0.55 + 32.70 * dh**2, 0.99)
    k = 1 - Z_TE
    return Y_p * k, Y_te * k, Y_s, Y_cl, Y_inc * k


_LOSS_NAMES = ("profile", "trailing", "secondary", "clearance", "incidence")


def benner_losses(
    c: _Cascade, f: dict, incidence_sign: str = "turboflow", delta_height_ratio: float = 0.011
) -> dict:
    """Loss coefficients of one row (stagnation-pressure definition), Benner et al. set, as a dict.

    `f` holds the flow quantities TurboFlow passes to its loss model: Re_in, Re_out, Ma_rel_in,
    Ma_rel_out, p0_rel_is, p_in, p0_rel_out, p_out, beta_in, beta_out, gamma_out.
    """
    sign = _incidence_sign(c, incidence_sign)
    y = _benner(c, f["Re_in"], f["Re_out"], f["Ma_rel_in"], f["Ma_rel_out"], f["p0_rel_is"], f["p_in"],
                f["p0_rel_out"], f["p_out"], f["beta_in"], f["beta_out"], f["gamma_out"], sign, delta_height_ratio)  # fmt: skip
    return {**dict(zip(_LOSS_NAMES, y, strict=True)), "total": sum(y)}


def _incidence_sign(c: _Cascade, mode: str) -> float:
    # TurboFlow 0.1.18 effectively uses -1 for every row ("turboflow"); its intended code gives +1 for
    # rotors ("rotor_plus"). See the module docstring for why "turboflow" is the default.
    return (1.0 if c.is_rotor else -1.0) if mode == "rotor_plus" else -1.0


def critical_mach(gamma: float, eta: float) -> float:
    """Critical Mach number of a nozzle with efficiency eta (TurboFlow `get_mach_crit`):
    Ma_crit^2 = 2/(g-1) (1/T_hat - 1), T_hat = (2a + eta - 3 + sqrt((1+eta)^2 + 4a(1+a-3 eta))) / (4a - 2),
    a = g/(g-1). This is the Mach number of maximum mass flow; Ma_crit = 1 for eta = 1."""
    a = gamma / (gamma - 1)
    T_hat = (2 * a + eta - 3 + math.sqrt((1 + eta) ** 2 + 4 * a * (1 + a - 3 * eta))) / (4 * a - 2)
    return math.sqrt(2 / (gamma - 1) * (1 / T_hat - 1))


def _aungier_angle(c: _Cascade, Ma_exit: float, Ma_crit: float) -> float:
    """Unsigned subsonic exit flow angle [deg] (Aungier 2006 deviation model, TurboFlow form):
    deviation delta0 (geometry) at Ma <= 0.5, blended to 0 at Ma_crit with 1 - 10X^3 + 15X^4 - 6X^5.
    """
    X = (Ma_exit - 0.5) / (Ma_crit - 0.5)
    p = 1.0 if X < 0 else (0.0 if X > 1 else 1 - X**3 * (10 - X * (15 - 6 * X)))
    return c.gauging - c.delta0 * p


# ======================================================================================
# Planes
# ======================================================================================
class _Plane:
    __slots__ = ("T", "p", "s", "h", "rho", "mu", "a", "gamma", "v", "vm", "vt", "alpha", "w", "wt", "beta",
                 "U", "h0", "h0rel", "p0", "T0", "p0rel", "Ma", "Ma_rel", "Re", "rothalpy", "losses", "h_is", "Y")  # fmt: skip

    def as_dict(self):
        return {k: getattr(self, k, None) for k in self.__slots__ if k != "losses"}


def _fill(gas: _Gas, pl: _Plane, chord: float):
    """Complete a plane whose static (T, p, s, h), velocities and U are known."""
    T = pl.T
    pl.rho = pl.p / (gas.R * T)
    pl.gamma = gas.gamma(T)
    pl.a = math.sqrt(pl.gamma * gas.R * T)
    pl.mu = air_viscosity(T)
    pl.Ma, pl.Ma_rel = pl.v / pl.a, pl.w / pl.a
    pl.Re = pl.rho * pl.w * chord / pl.mu
    pl.h0 = pl.h + 0.5 * pl.v * pl.v
    pl.h0rel = pl.h + 0.5 * pl.w * pl.w
    pl.rothalpy = pl.h0rel - 0.5 * pl.U * pl.U


def _stagnation(gas: _Gas, pl: _Plane):
    """Absolute and relative stagnation T/p (for output)."""
    pl.T0 = gas.T_h(pl.h0, pl.T)
    pl.p0 = gas.p_Ts(pl.T0, pl.s)
    pl.p0rel = gas.p_Ts(gas.T_h(pl.h0rel, pl.T), pl.s)


def _from_velocity_in(gas, c: _Cascade, h0, s, v, alpha, omega, Tg):
    """Inlet plane from absolute velocity and angle (TurboFlow evaluate_cascade_inlet)."""
    pl = _Plane()
    pl.U = omega * c.r_in
    pl.v, pl.alpha = v, alpha
    pl.vm, pl.vt = v * _cosd(alpha), v * _sind(alpha)
    pl.wt = pl.vt - pl.U
    pl.w = math.hypot(pl.vm, pl.wt)
    pl.beta = _atand(pl.wt / pl.vm)
    pl.h = h0 - 0.5 * v * v
    pl.s = s
    pl.T = gas.T_h(pl.h, Tg)
    pl.p = gas.p_Ts(pl.T, s)
    _fill(gas, pl, c.chord)
    pl.p0rel = gas.p_Ts(gas.T_h(pl.h0rel, pl.T), s)
    return pl


def _solve_velocity(gas, h0f, s, area_cos, mdot, v_guess, Tg, T0f=None):
    """Subsonic velocity with rho v area_cos = mdot along the isentrope s, stagnation enthalpy h0f.

    Solved in temperature (v = sqrt(2 (h0f - h(T))), p = p(T, s), rho = p / RT), which avoids an inner
    h -> T inversion. Safeguarded Newton on [T_lo, T_hi]: the flux rises from 0 at T0 to its maximum
    at the sonic temperature; if the bracket collapses onto the sonic point the row is choked (raises
    _Choke). d ln(flux)/dT = (cp/R - 1)/T - cp/v^2. Returns (v, T, p)."""
    R = gas.R
    if T0f is None:
        T0f = gas.T_h(h0f, Tg)
    T_hi, T_lo = T0f, max(60.0, 0.2 * T0f)
    if v_guess > 0:
        T = min(max(Tg, T_lo + 1e-6 * T0f), T_hi * (1 - 1e-9))
    else:
        T = T0f * (1 - 1e-3)
    for _ in range(100):
        h = gas.h(T)
        v2 = 2.0 * (h0f - h)
        if v2 <= 0.0:
            T_hi = T
            T = 0.5 * (T_lo + T_hi)
            continue
        cp = gas.cp(T)
        v = math.sqrt(v2)
        p = gas.p_Ts(T, s)
        flux = p / (R * T) * v * area_cos
        M2 = v2 * (cp - R) / (cp * R * T)
        F = flux - mdot
        if M2 >= 1.0:
            T_lo = T
            T_new = 0.5 * (T_lo + T_hi)
        else:
            if F < 0:
                T_hi = T
            else:
                T_lo = T
            dF = flux * ((cp / R - 1.0) / T - cp / v2)
            T_new = T - F / dF
            if not T_lo < T_new < T_hi:
                T_new = 0.5 * (T_lo + T_hi)
        if abs(T_new - T) < 1e-12 * T:
            if M2 >= 1.0 or abs(F) > 1e-8 * mdot:
                raise _Choke(
                    "no subsonic solution (flux maximum at the sonic point is below the demand)"
                )
            return v, T, p
        T = T_new
    raise _Choke("continuity did not converge (near choke)")


# ======================================================================================
# Scalar root finder for monotone functions with an unknown feasibility limit
# ======================================================================================
_FAIL = (ArithmeticError, ValueError)  # plus _Choke / _LimitLoading (subclasses of _ModelLimit)


class _Root:
    __slots__ = ("status", "x", "v", "payload", "x_fail", "exc")

    def __init__(self, status, x, v, payload, x_fail=None, exc=None):
        self.status, self.x, self.v, self.payload, self.x_fail, self.exc = (
            status,
            x,
            v,
            payload,
            x_fail,
            exc,
        )


def _monotone_root(f, x0, dx, increasing, fail_high, xtol, vtol, x_max=None, ltol=None, maxit=100):
    """Root of a monotone scalar function f(x) -> (value, payload).

    f may raise _Choke / _LimitLoading beyond an unknown feasibility limit on one side (fail_high:
    large x fails). Secant extrapolation until the root is bracketed, then Illinois regula falsi;
    a failed step is bisected back toward the last feasible point.
    Returns _Root with status "root", "limit" (the root lies beyond the feasibility limit, located to
    within ltol; x/payload = closest feasible point, exc = the failure) or "bound" (root beyond x_max;
    x = x_max)."""
    ltol = xtol if ltol is None else ltol
    s = 1.0 if increasing else -1.0
    lo = hi = None  # (x, u, payload) with u = s * f < 0 (lo) and > 0 (hi)
    last = prev = None
    x_fail, exc = None, None
    side = 0  # Illinois bookkeeping
    x = min(x0, x_max) if x_max is not None else x0
    for _ in range(maxit):
        try:
            v, pay = f(x)
        except (_ModelLimit, *_FAIL) as e:
            if x_fail is None or (fail_high and x < x_fail) or (not fail_high and x > x_fail):
                x_fail, exc = x, e
            oks = [p for p in (lo, hi) if p is not None]
            if not oks:
                x = (
                    x - dx if fail_high else x + dx
                )  # nothing feasible yet: step away from the limit
                dx *= 2
                continue
            ok = max(oks, key=lambda p: p[0]) if fail_high else min(oks, key=lambda p: p[0])
            if abs(x_fail - ok[0]) < ltol:
                return _Root("limit", ok[0], ok[1] * s, ok[2], x_fail, exc)
            x = 0.5 * (ok[0] + x_fail)
            continue
        u = s * v
        if abs(v) <= vtol:
            return _Root("root", x, v, pay)
        pt = (x, u, pay)
        if u < 0:
            lo = pt
        else:
            hi = pt
        prev, last = last, pt
        if lo is not None and hi is not None:
            if abs(hi[0] - lo[0]) < xtol:
                best = lo if abs(lo[1]) < abs(hi[1]) else hi
                return _Root("root", best[0], best[1] * s, best[2])
            # Illinois: halve the retained end's value when the same end is kept twice
            ul, uh = lo[1], hi[1]
            if u < 0:
                if side == -1:
                    uh *= 0.5
                side = -1
            else:
                if side == 1:
                    ul *= 0.5
                side = 1
            x = lo[0] - ul * (hi[0] - lo[0]) / (uh - ul)
            continue
        # not bracketed yet: secant extrapolation (or a fixed step) toward the root
        step_dir = 1.0 if u < 0 else -1.0  # u increasing in x
        if prev is not None and (last[1] - prev[1]) * (last[0] - prev[0]) > 0:
            x_new = last[0] - last[1] * (last[0] - prev[0]) / (last[1] - prev[1])
            lim = 4 * abs(last[0] - prev[0]) + abs(dx)
            x_new = min(max(x_new, last[0] - lim), last[0] + lim)
        else:
            x_new = last[0] + step_dir * abs(dx)
        if x_fail is not None and (
            (fail_high and x_new >= x_fail) or (not fail_high and x_new <= x_fail)
        ):
            if abs(x_fail - last[0]) < ltol:
                return _Root("limit", last[0], last[1] * s, last[2], x_fail, exc)
            x_new = 0.5 * (last[0] + x_fail)
        if x_max is not None and x_new > x_max:
            if last[0] >= x_max:
                return _Root("bound", last[0], last[1] * s, last[2])
            x_new = x_max
        x = x_new
    raise _Choke("root finder did not converge")


# ======================================================================================
# Turbine
# ======================================================================================
class AxialTurbineMeanline:
    """Fast single-stage axial turbine (stator + rotor) meanline model.

    >>> turb = AxialTurbineMeanline(AxialStageGeometry.from_turboflow_yaml(KOFSKEY_1972_CONFIG))
    >>> pt = turb.evaluate(T0_in_K=295.6, p0_in_Pa=1.38e5, omega_rad_s=1627.0, pr_ts=2.298)
    >>> pt.mass_flow_kg_s, pt.eta_ts, pt.power_W

    warm_start: reuse the previous solution's row states as initial guesses (much faster in sweeps and
    engine matching; results change only at the solver-tolerance level, ~1e-9).
    loss_multiplier: k_loss, multiplies every row's total loss coefficient Y (Benner profile + trailing edge +
    secondary + clearance + incidence). 1.0 = the published correlations unchanged. A value fitted to measured
    efficiency is an EFFECTIVE calibration factor (like the compressor's diffuser Cf): it absorbs whatever the
    correlations miss at micro-turbine scale (low Reynolds number, relative roughness, clearance, leakage).
    """

    def __init__(self, geometry: AxialStageGeometry, gas: GasMixture | None = None,
                 incidence_sign: str = "turboflow", blockage: float = 0.0,
                 delta_height_ratio: float = 0.011, alpha_in_deg: float = 0.0, tol: float = 1e-9,
                 warm_start: bool = True, loss_multiplier: float = 1.0):  # fmt: skip
        incidence_sign = {"corrected": "rotor_plus"}.get(incidence_sign, incidence_sign)  # old name
        if incidence_sign not in ("turboflow", "rotor_plus"):
            raise ValueError("incidence_sign must be 'turboflow' or 'rotor_plus'")
        self.geometry = geometry
        self.c = geometry.cascades()
        self._gas_mixture = gas if gas is not None else air()
        self.gas = _Gas(self._gas_mixture)
        self.incidence_sign, self.blockage = incidence_sign, blockage
        self.dh_ratio, self.alpha_in, self.tol = delta_height_ratio, alpha_in_deg, tol
        self._sign = [_incidence_sign(c, incidence_sign) for c in self.c]
        self.warm_start = warm_start
        self.loss_multiplier = float(loss_multiplier)
        self._ws: dict = {}

    def set_gas(self, gas: GasMixture) -> None:
        """Change the working gas in place (keeps the warm-start state; used by the engine solver)."""
        if gas is not self._gas_mixture:
            self._gas_mixture, self.gas = gas, _Gas(gas)

    def with_gas(self, gas: GasMixture) -> AxialTurbineMeanline:
        return AxialTurbineMeanline(self.geometry, gas, self.incidence_sign, self.blockage, self.dh_ratio,
                                    self.alpha_in, self.tol, self.warm_start, self.loss_multiplier)  # fmt: skip

    # ---------------------------------------------------------------- row pieces
    def _Y(self, c, inlet, pl, p0is):
        y = _benner(c, inlet.Re, pl.Re, inlet.Ma_rel, pl.Ma_rel, p0is, inlet.p, pl.p0rel, pl.p, inlet.beta,
                    pl.beta, pl.gamma, self._sign[c.index], self.dh_ratio)  # fmt: skip
        return y if self.loss_multiplier == 1.0 else tuple(self.loss_multiplier * v for v in y)

    def _outlet_plane(self, c, r, w, beta, T, p, s, omega):
        pl = _Plane()
        pl.U = omega * r
        pl.w, pl.beta = w, beta
        sb, cb = _sind(beta), _cosd(beta)
        pl.wt, pl.vm = w * sb, w * cb
        pl.vt = pl.wt + pl.U
        pl.v = math.hypot(pl.vm, pl.vt)
        pl.alpha = _atand(pl.vt / pl.vm) if pl.vm > 0 else 90.0
        pl.T, pl.p, pl.s = T, p, s
        pl.h = self.gas.h(T)
        _fill(self.gas, pl, c.chord)
        return pl

    def _row_out(self, c, inlet, mdot, omega, where, Ma_crit=None, p_exit=None):
        """Throat or exit plane of a row with mass flow mdot.

        where = "throat": beta = gauging angle, subsonic continuity (throat area normal to the flow).
        where = "exit", p_exit None: subsonic continuity, Aungier deviation with Ma_crit.
        where = "exit", p_exit given: choked row; static pressure given, beta from continuity.
        Loss and continuity are iterated together (secant-accelerated fixed point on the entropy).
        """
        gas = self.gas
        omega = omega if c.is_rotor else 0.0
        r = c.r_throat if where == "throat" else c.r_out
        area = (c.A_throat if where == "throat" else c.A_out) * (1 - self.blockage)
        U = omega * r
        h0rel = inlet.rothalpy + 0.5 * U * U
        T0rel = gas.T_h(h0rel, inlet.T)
        p0is = gas.p_Ts(T0rel, inlet.s)  # relative stagnation pressure of an isentropic row
        key = (c.index, where, p_exit is not None)
        ws = self._ws.get(key)
        if ws is not None:
            s, beta, w = inlet.s + ws[0], ws[1], ws[2]
        else:
            s, beta = inlet.s + 1e-3 * gas.R, c.sign * c.gauging
            w = 0.6 * math.sqrt(gas.gamma(T0rel) * gas.R * T0rel)
        if p_exit is None:
            pl = self._row_out_fused(
                c, inlet, mdot, omega, where, Ma_crit, r, area, h0rel, T0rel, p0is, s, beta, w, key
            )
            if pl is not None:
                return pl
        T = T0rel
        prev = good = None
        backoff = 0
        R = gas.R
        for _ in range(150):
            if p_exit is None:
                a_eff = area if where == "throat" else area * _cosd(beta)
                try:
                    w, T, p = _solve_velocity(gas, h0rel, s, a_eff, mdot, w, T, T0rel)
                except _Choke:
                    # a secant/deviation step overshot into an entropy (or angle) that can't pass the
                    # flow: step back halfway toward the last state that could
                    backoff += 1
                    if good is None or backoff > 30:
                        raise
                    s, beta = 0.5 * (s + good[0]), 0.5 * (beta + good[1])
                    prev = None
                    continue
                good = (s, beta)
            else:
                p = p_exit
                T = gas.T_ps(p, s, T)
                dh = h0rel - gas.h(T)
                if dh <= 0:
                    raise _LimitLoading("exit pressure above the relative stagnation pressure")
                w = math.sqrt(2 * dh)
                cb = mdot * R * T / (p * w * area)
                if cb > 1.0:
                    raise _LimitLoading(
                        "cos(beta) > 1: the exit cannot pass the flow at this pressure"
                    )
                beta = c.sign * _acosd(cb)
            pl = self._outlet_plane(c, r, w, beta, T, p, s, omega)
            pl.p0rel = p0is * math.exp(-(s - inlet.s) / R)
            y = self._Y(c, inlet, pl, p0is)
            Y = y[0] + y[1] + y[2] + y[3] + y[4]
            p0rel_new = (p0is + Y * p) / (1 + Y)
            s_new = inlet.s + R * math.log(p0is / p0rel_new)
            beta_new = (
                c.sign * _aungier_angle(c, pl.Ma_rel, Ma_crit)
                if (where == "exit" and p_exit is None)
                else beta
            )
            F = s_new - s
            done = abs(F) < self.tol * R and abs(beta_new - beta) < 1e-8
            if prev is not None and abs(F - prev[1]) > 1e-14 * R:
                s_next = max(inlet.s, s - F * (s - prev[0]) / (F - prev[1]))
            else:
                s_next = s_new
            prev = (s, F)
            s, beta = s_next, beta_new
            if done:
                break
        else:
            raise _Choke(f"{c.type} {where}: loss iteration did not converge")
        self._ws[key] = (s - inlet.s, beta, w)
        pl.losses, pl.Y = y, Y
        pl.p0rel = p0rel_new
        return pl

    def _row_out_fused(
        self, c, inlet, mdot, omega, where, Ma_crit, r, area, h0rel, T0rel, p0is, s, beta, w, key
    ):
        """Fast path of `_row_out` (subsonic continuity).

        Unknowns: static temperature T and entropy s; at the exit the flow angle follows the Aungier
        deviation model implicitly, beta = beta(Ma(T)). Each iteration evaluates the plane, updates s
        from the loss model (secant-accelerated fixed point) and takes a Newton step on T for
        continuity that includes the entropy update and d beta/dT:
            d ln(flux) = [(cp/R - 1)/T - cp/v^2 - tan(beta) dbeta/dT] dT - ds/R.
        Returns None if it doesn't converge cleanly (e.g. near choke); the caller then uses the robust
        nested iteration, which also decides whether the row is choked."""
        gas, R = self.gas, self.gas.R
        exit_ = where == "exit"
        ws = self._ws.get((key, "T"))
        T = ws * T0rel if ws is not None else T0rel - 0.5 * w * w / gas.cp(T0rel)
        prev = None
        for _ in range(40):
            h = gas.h(T)
            v2 = 2.0 * (h0rel - h)
            if v2 <= 0.0:
                return None
            cp = gas.cp(T)
            v = math.sqrt(v2)
            p = gas.p_Ts(T, s)
            gam = cp / (cp - R)
            Ma = v / math.sqrt(gam * R * T)
            if Ma >= 0.99:
                return None
            dbeta_dT = 0.0
            if exit_:
                X = (Ma - 0.5) / (Ma_crit - 0.5)
                if 0.0 < X < 1.0:
                    pX = 1 - X**3 * (10 - X * (15 - 6 * X))
                    dbeta_dMa = (
                        30.0 * c.delta0 * X * X * (1 - X) ** 2 / (Ma_crit - 0.5)
                    )  # deg, unsigned
                    dbeta_dT = c.sign * dbeta_dMa * Ma * (-cp / v2 - 0.5 / T)
                else:
                    pX = 1.0 if X <= 0.0 else 0.0
                beta = c.sign * (c.gauging - c.delta0 * pX)
                cb = _cosd(beta)
            else:
                cb = 1.0
            rho = p / (R * T)
            flux = rho * v * area * cb
            p0rel = p0is * math.exp(-(s - inlet.s) / R)
            y = _benner(c, inlet.Re, rho * v * c.chord / air_viscosity(T), inlet.Ma_rel, Ma, p0is, inlet.p, p0rel,
                        p, inlet.beta, beta, gam, self._sign[c.index], self.dh_ratio)  # fmt: skip
            if self.loss_multiplier != 1.0:
                y = tuple(self.loss_multiplier * v for v in y)
            Y = y[0] + y[1] + y[2] + y[3] + y[4]
            p0rel_new = (p0is + Y * p) / (1 + Y)
            s_new = inlet.s + R * math.log(p0is / p0rel_new)
            F = s_new - s
            if prev is not None and abs(F - prev[1]) > 1e-14 * R:
                s_next = max(inlet.s, s - F * (s - prev[0]) / (F - prev[1]))
            else:
                s_next = s_new
            prev = (s, F)
            a = (cp / R - 1.0) / T - cp / v2 - math.tan(beta * _D2R) * dbeta_dT * _D2R
            if a >= 0.0:
                return None
            dT = ((mdot - flux) / flux + (s_next - s) / R) / a
            if abs(dT) > 0.05 * T:
                dT = math.copysign(0.05 * T, dT)
            if abs(dT) < 1e-10 * T and abs(F) < self.tol * R:
                self._ws[key] = (s - inlet.s, beta, v)
                self._ws[(key, "T")] = T / T0rel
                pl = self._outlet_plane(c, r, v, beta, T, p, s, omega)
                pl.losses, pl.Y, pl.p0rel = y, Y, p0rel_new
                return pl
            T, s = T + dT, s_next
        return None

    def _throat_crit(self, c, inlet, omega):
        """Critical (choking) throat state for a given inlet plane: throat Ma_rel = Ma_crit(eta)."""
        gas = self.gas
        omega = omega if c.is_rotor else 0.0
        U = omega * c.r_throat
        h0rel = inlet.rothalpy + 0.5 * U * U
        T0rel = gas.T_h(h0rel, inlet.T)
        p0is = gas.p_Ts(T0rel, inlet.s)
        area = c.A_throat * (1 - self.blockage)
        key = (c.index, "crit")
        ds, M = self._ws.get(key, (0.0, 0.95))
        s = inlet.s + ds
        T = T0rel / (1 + 0.2 * M * M)
        R = gas.R
        for _ in range(100):
            for _k in range(30):  # T with h(T) + M^2 gamma R T / 2 = h0rel
                g = gas.gamma(T)
                F = gas.h(T) + 0.5 * M * M * g * R * T - h0rel
                dT = F / (gas.cp(T) + 0.5 * M * M * g * R)
                T -= dT
                if abs(dT) < 1e-11 * T:
                    break
            w = M * math.sqrt(gas.gamma(T) * R * T)
            p = gas.p_Ts(T, s)
            pl = self._outlet_plane(c, c.r_throat, w, c.sign * c.gauging, T, p, s, omega)
            pl.p0rel = p0is * math.exp(-(s - inlet.s) / R)
            y = self._Y(c, inlet, pl, p0is)
            Y = sum(y)
            s_new = inlet.s + R * math.log(p0is / ((p0is + Y * p) / (1 + Y)))
            h_is = gas.h(gas.T_ps(p, inlet.s, T))
            M_new = critical_mach(pl.gamma, (h0rel - pl.h) / (h0rel - h_is))
            done = abs(s_new - s) < self.tol * R and abs(M_new - M) < 1e-11
            s, M = s_new, M_new
            if done:
                break
        else:
            raise _Choke(f"{c.type}: critical throat state did not converge")
        self._ws[key] = (s - inlet.s, M)
        pl.losses, pl.Y = y, Y
        return pl.rho * w * area, M, pl

    def _crit_mach_of(self, pl: _Plane, h0rel: float, s_in: float) -> float:
        h_is = self.gas.h(self.gas.T_ps(pl.p, s_in, pl.T))
        return critical_mach(pl.gamma, (h0rel - pl.h) / (h0rel - h_is))

    def _interspace(self, c_prev, ex: _Plane, c_next):
        """Gap between rows (TurboFlow evaluate_cascade_interspace): h0 and r*C_theta conserved, rho constant."""
        vt = ex.vt * c_prev.r_out / c_next.r_in
        vm = ex.vm * c_prev.A_out / c_next.A_in * (1 - self.blockage)
        v = math.hypot(vt, vm)
        T = self.gas.T_h(ex.h0 - 0.5 * v * v, ex.T)
        p = ex.rho * self.gas.R * T
        return ex.h0, self.gas.s_Tp(T, p), _atand(vt / vm), v

    def _stator_inlet(self, h0, s, mdot, T0):
        c = self.c[0]
        vg, Tg = self._ws.get("in", (0.0, T0))
        v, T, _ = _solve_velocity(self.gas, h0, s, c.A_in * _cosd(self.alpha_in), mdot, vg, Tg, T0)
        self._ws["in"] = (v, T)
        return _from_velocity_in(self.gas, c, h0, s, v, self.alpha_in, 0.0, T)

    def _row_subsonic(self, c, inlet, mdot, omega):
        th = self._row_out(c, inlet, mdot, omega, "throat")
        U = omega * c.r_throat if c.is_rotor else 0.0
        Ma_crit = self._crit_mach_of(th, inlet.rothalpy + 0.5 * U * U, inlet.s)
        ex = self._row_out(c, inlet, mdot, omega, "exit", Ma_crit=Ma_crit)
        return th, ex, Ma_crit

    def _rotor_inlet(self, ex0, omega):
        h0, s, alpha, v = self._interspace(self.c[0], ex0, self.c[1])
        return _from_velocity_in(self.gas, self.c[1], h0, s, v, alpha, omega, ex0.T)

    # ---------------------------------------------------------------- public API
    def evaluate(
        self, T0_in_K: float, p0_in_Pa: float, omega_rad_s: float, pr_ts: float
    ) -> TurbinePoint:
        """Operating point at total-to-static pressure ratio pr_ts = p0_in / p_out."""
        pt = TurbinePoint(T0_in_K=T0_in_K, p0_in_Pa=p0_in_Pa, omega_rad_s=omega_rad_s, pr_ts=pr_ts)
        if not (pr_ts > 1.0 and T0_in_K > 0 and p0_in_Pa > 0 and omega_rad_s >= 0):
            pt.failure_reason = "invalid_input"
            return pt
        if not self.warm_start:
            self._ws = {}
        ws_backup = dict(self._ws)
        try:
            self._solve(pt)
            pt.converged = True
        except _LimitLoading as e:
            pt.failure_reason, pt.message = "limit_loading", str(e)
        except (_Choke, *_FAIL) as e:
            pt.failure_reason, pt.message = "no_convergence", str(e)
        if not pt.converged:
            self._ws = ws_backup  # don't let a failed point spoil the next initial guess
        return pt

    def evaluate_many(self, points) -> list[TurbinePoint]:
        """Evaluate (T0_in_K, p0_in_Pa, omega_rad_s, pr_ts) tuples in order (warm-started)."""
        return [self.evaluate(*p) for p in points]

    def _solve(self, pt: TurbinePoint):
        gas, omega = self.gas, pt.omega_rad_s
        T0, p0 = pt.T0_in_K, pt.p0_in_Pa
        h0, s_in = gas.h(T0), gas.s_Tp(T0, p0)
        p_target = p0 / pt.pr_ts
        st, ro = self.c
        vtol = 1e-9 * p0  # exit-pressure tolerance (relative to p0_in)
        g = gas.gamma(T0)
        # reference: isentropic choked flow through the stator throat (scales the root finding)
        m_ref = (
            st.A_throat
            * p0
            / math.sqrt(gas.R * T0)
            * math.sqrt(g)
            * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        )
        hint = self._ws.setdefault("hint", {})

        def march(m):
            try:
                i0 = self._stator_inlet(h0, s_in, m, T0)
                th0, ex0, _ = self._row_subsonic(st, i0, m, 0.0)
            except _Choke as e:
                e.row = 0
                raise
            try:
                i1 = self._rotor_inlet(ex0, omega)
                th1, ex1, _ = self._row_subsonic(ro, i1, m, omega)
            except _Choke as e:
                e.row = 1
                raise
            return [i0, th0, ex0, i1, th1, ex1]

        def f_A(m):
            planes = march(m)
            return planes[-1].p - p_target, planes

        def regime_A(x0, ltol, x_max=None, dx=0.02):
            return _monotone_root(f_A, x0, dx * m_ref, increasing=False, fail_high=True,
                                  xtol=1e-10 * m_ref, vtol=vtol, ltol=ltol, x_max=x_max)  # fmt: skip

        def done_A(r):
            hint["mA"], hint["regime"] = r.x / m_ref, "A"
            return self._result(pt, r.payload, r.x, "")

        # ---- quick path when the previous point was subsonic: regime A first (coarse limit search)
        m_guess = None
        if hint.get("regime") == "A":
            rA = regime_A(hint["mA"] * m_ref, 2e-3 * m_ref, dx=0.003)
            if rA.status == "root":
                return done_A(rA)
            m_guess = rA.x

        # ---- choke structure. Stator critical flow: fixed point with the inlet continuity
        m = m_guess if m_guess is not None else hint.get("ms", 0.98) * m_ref
        for _ in range(60):
            i0 = self._stator_inlet(h0, s_in, m, T0)
            m_new, mc0, th0 = self._throat_crit(st, i0, omega)
            if abs(m_new - m) < 1e-11 * m_new:
                break
            m = m_new
        else:
            raise _Choke("stator critical flow did not converge")
        m_s = m_new
        hint["ms"] = m_s / m_ref

        def g_rotor(m):  # rotor critical flow minus m, with a subsonic stator
            i0 = self._stator_inlet(h0, s_in, m, T0)
            th0, ex0, _ = self._row_subsonic(st, i0, m, 0.0)
            i1 = self._rotor_inlet(ex0, omega)
            m_c, mc1, th1 = self._throat_crit(ro, i1, omega)
            return m_c - m, (i0, th0, ex0, i1, th1, mc1)

        # which row chokes first? the rotor, if it is already critical below the stator's critical flow
        rotor_first, m_test = None, m_s * (1 - 1e-9)
        for _ in range(8):
            try:
                rotor_first = g_rotor(m_test)[0] < 0
                break
            except (
                _ModelLimit
            ):  # stator exit can't pass this flow subsonically: test a little lower
                m_test *= 0.997
        if rotor_first is None:
            raise _Choke("could not determine which row chokes first")

        if rotor_first:
            # ---- rotor chokes first: m = rotor critical flow with a subsonic stator
            x0 = min(hint.get("mr", 0.995 * m_s / m_ref) * m_ref, m_test)
            rR = _monotone_root(g_rotor, x0, 2e-3 * m_ref, increasing=False, fail_high=True,
                                xtol=1e-11 * m_ref, vtol=1e-11 * m_ref, ltol=1e-6 * m_ref)  # fmt: skip
            if rR.status == "root":
                i0, th0, ex0, i1, th1, mc1 = rR.payload
                m_r = rR.x
                hint["mr"] = m_r / m_ref
                try:  # choked-branch onset: exit pressure where the subsonic exit reaches Ma_crit
                    p_on = self._row_out(ro, i1, m_r, omega, "exit", Ma_crit=mc1).p
                except _Choke:
                    p_on = th1.p
                if p_target > p_on:  # subsonic: regime A, bracketed by the rotor critical flow
                    rA = regime_A(
                        min(hint.get("mA", 0.98 * m_r / m_ref) * m_ref, m_r),
                        1e-9 * m_ref,
                        x_max=m_r,
                    )
                    if rA.status == "root":
                        return done_A(rA)
                    # otherwise in the (< ~0.3 % mass-flow) gap between the branches: choked branch
                ex1 = self._row_out(ro, i1, m_r, omega, "exit", p_exit=p_target)
                hint["regime"] = "D"
                return self._result(pt, [i0, th0, ex0, i1, th1, ex1], m_r, "rotor")
            # the stator can't pass the rotor's critical flow: the stator chokes first after all

        # ---- stator chokes first: m = m_s
        i0 = self._stator_inlet(h0, s_in, m_s, T0)
        _, mc0, th0 = self._throat_crit(st, i0, omega)
        try:
            p1_on = self._row_out(st, i0, m_s, 0.0, "exit", Ma_crit=mc0).p
        except _Choke:
            p1_on = (
                th0.p
            )  # the exit can't pass the critical flow subsonically: choked branch from the throat pressure

        def stator_exit(p1):
            return self._row_out(st, i0, m_s, 0.0, "exit", p_exit=p1)

        def f_B(p1):  # regime B: stator choked, rotor subsonic
            ex0 = stator_exit(p1)
            try:
                i1 = self._rotor_inlet(ex0, omega)
                th1, ex1, _ = self._row_subsonic(ro, i1, m_s, omega)
            except _Choke as e:
                e.row = 1
                raise
            return ex1.p - p_target, [i0, th0, ex0, i1, th1, ex1]

        try:
            p_on = f_B(p1_on)[0] + p_target
        except _ModelLimit:
            p_on = -1.0  # rotor can't pass the flow at the onset: choked analysis below
        if p_target > p_on:  # subsonic: regime A, bracketed by the stator critical flow
            rA = regime_A(
                min(hint.get("mA", 0.98 * m_s / m_ref) * m_ref, m_s),
                1e-9 * m_ref,
                x_max=m_s * (1 - 1e-9),
            )
            if rA.status == "root":
                return done_A(rA)

        rB = _monotone_root(f_B, min(hint.get("p1", 1.0) * p0, p1_on), 0.02 * p1_on, increasing=True,
                            fail_high=False, xtol=1e-10 * p0, vtol=vtol, x_max=0.999 * p0, ltol=1e-6 * p0)  # fmt: skip
        if rB.status == "root":
            # (rB.x > p1_on: target in the < ~0.3 % mass-flow gap between subsonic and choked branch)
            hint["p1"], hint["regime"] = rB.x / p0, "B"
            return self._result(pt, rB.payload, m_s, "stator")
        if rB.status == "bound":
            raise _Choke("stator-choked branch does not reach the target pressure")
        if isinstance(rB.exc, _LimitLoading) or getattr(rB.exc, "row", 0) != 1:
            raise _LimitLoading("stator limit loading before the target pressure ratio")

        # ---- regime C: both rows choked; p1 such that the rotor is exactly critical
        def margin(p1):
            ex0 = stator_exit(p1)
            i1 = self._rotor_inlet(ex0, omega)
            m_c, _, th1 = self._throat_crit(ro, i1, omega)
            return m_c - m_s, (ex0, i1, th1)

        x0 = hint.get("p1c", rB.x / p0) * p0
        rC = _monotone_root(margin, x0, 0.5 * abs(rB.x - rB.x_fail) + 1e-6 * p0, increasing=True,
                            fail_high=False, xtol=1e-11 * p0, vtol=1e-11 * m_s)  # fmt: skip
        if rC.status != "root":
            raise _LimitLoading("stator limit loading before the rotor chokes")
        ex0, i1, th1 = rC.payload
        ex1 = self._row_out(ro, i1, m_s, omega, "exit", p_exit=p_target)
        hint["p1c"], hint["regime"] = rC.x / p0, "C"
        return self._result(pt, [i0, th0, ex0, i1, th1, ex1], m_s, "stator+rotor")

    def _result(self, pt, planes, mdot, choked):
        gas = self.gas
        i0, th0, ex0, i1, th1, ex1 = planes
        h0 = gas.h(pt.T0_in_K)
        s_in = gas.s_Tp(pt.T0_in_K, pt.p0_in_Pa)
        for pl in planes:
            _stagnation(gas, pl)
        h_out_s = gas.h(gas.T_ps(ex1.p, s_in, ex1.T))
        dh0 = h0 - ex1.h0
        pt.mass_flow_kg_s = mdot
        pt.work_J_kg = dh0
        pt.power_W = mdot * dh0
        pt.torque_N_m = pt.power_W / pt.omega_rad_s if pt.omega_rad_s > 0 else float("nan")
        pt.eta_ts = dh0 / (h0 - h_out_s)
        pt.eta_tt = dh0 / (h0 - h_out_s - 0.5 * ex1.v**2)
        pt.T0_out_K, pt.p0_out_Pa, pt.p_out_Pa, pt.T_out_K = ex1.T0, ex1.p0, ex1.p, ex1.T
        pt.exit_flow_angle_deg = ex1.alpha
        pt.exit_mach = ex1.Ma
        pt.reaction = (ex0.h - ex1.h) / (i0.h - ex1.h)
        pt.choked = choked
        pt.stator_exit_mach = ex0.Ma
        pt.rotor_exit_mach_rel = ex1.Ma_rel
        pt.rotor_inlet_beta_deg = i1.beta
        pt.rotor_incidence_deg = i1.beta - self.c[1].theta_in
        pt.T0_rel_rotor_in_K = gas.T_h(i1.h0rel, i1.T)
        pt.u_mean_m_s = self.c[1].r_out * pt.omega_rad_s
        pt.losses = {"stator": dict(zip(_LOSS_NAMES, ex0.losses, strict=True)),
                     "rotor": dict(zip(_LOSS_NAMES, ex1.losses, strict=True))}  # fmt: skip
        pt.planes = [pl.as_dict() for pl in planes]
        pt.p_error_rel = (ex1.p - pt.p0_in_Pa / pt.pr_ts) / pt.p0_in_Pa
        if (
            not abs(pt.p_error_rel) < 1e-7
        ):  # safety net: never return a point that misses the target
            raise _Choke(
                f"exit pressure misses the target by {pt.p_error_rel:.2e} (relative to p0_in)"
            )


@dataclass
class TurbinePoint:
    """One turbine operating point (SI). Generic: an axial or (later) radial turbine fills the same fields."""

    T0_in_K: float
    p0_in_Pa: float
    omega_rad_s: float
    pr_ts: float
    converged: bool = False
    failure_reason: str = ""
    message: str = ""
    mass_flow_kg_s: float = float("nan")
    work_J_kg: float = float("nan")
    power_W: float = float("nan")
    torque_N_m: float = float("nan")
    eta_ts: float = float("nan")
    eta_tt: float = float("nan")
    T0_out_K: float = float("nan")
    p0_out_Pa: float = float("nan")
    p_out_Pa: float = float("nan")
    T_out_K: float = float("nan")
    exit_flow_angle_deg: float = float("nan")
    exit_mach: float = float("nan")
    reaction: float = float("nan")
    choked: str = ""  # "", "stator", "rotor", "stator+rotor"
    stator_exit_mach: float = float("nan")
    rotor_exit_mach_rel: float = float("nan")
    rotor_inlet_beta_deg: float = float("nan")
    rotor_incidence_deg: float = float("nan")
    T0_rel_rotor_in_K: float = float("nan")
    u_mean_m_s: float = float("nan")
    p_error_rel: float = float("nan")
    losses: dict = field(default_factory=dict, repr=False)
    planes: list = field(default_factory=list, repr=False)

    def scalars(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("losses", "planes")}


class TurbineModel(Protocol):
    """Interface the engine model uses; an axial or a (future) radial turbine implements it."""

    def evaluate(
        self, T0_in_K: float, p0_in_Pa: float, omega_rad_s: float, pr_ts: float
    ) -> TurbinePoint: ...

    def with_gas(self, gas: GasMixture) -> TurbineModel: ...

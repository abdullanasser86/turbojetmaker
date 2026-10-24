"""Independent 0-D design-point turbojet cycle (pure NumPy), for cross-checking pyCycle.

What a turbojet cycle calculation does, step by step
----------------------------------------------------
We follow one kilogram of air through the engine, using *total* (stagnation) quantities
and the gas model in :mod:`turboml.gas.thermo`.

0 → 2  Intake. Free-stream static state (Ts0, Ps0) plus flight speed V0 gives totals:
       h(Tt0) = h(Ts0) + V0²/2, Pt0 from the isentropic relation.
       Losses: Pt2 = π_d · Pt0, Tt2 = Tt0 (adiabatic).
2 → 3  Compressor with pressure ratio PR_c and isentropic efficiency η_c:
       T3s from isentropic compression to Pt3 = PR_c · Pt2;
       h3 = h2 + (h(T3s) − h2) / η_c;   power P_c = W (h3 − h2).
3 → 4  Combustor. Energy balance with fuel-air ratio f (absolute enthalpies):
       h_air(Tt3) + f · h_fuel,eff = (1 + f) · h_prod(f, Tt4),
       where h_fuel,eff includes the LHV and the combustion efficiency
       (see :func:`turboml.gas.thermo.fuel_enthalpy_J_per_kg`). Solved for f.
       Pressure loss: Pt4 = (1 − ΔP/P) · Pt3.
4 → 5  Turbine drives the compressor (shaft power balance):
       η_m · P_t = P_c + P_offtake,  P_t = W (1 + f)(h4 − h5)  →  h5, Tt5.
       Turbine pressure ratio from the isentropic efficiency:
       h5s = h4 − (h4 − h5) / η_t,  Pt4/Pt5 from the isentropic relation T4 → T5s.
5 → 9  Convergent nozzle (Pt9 = (1 − ΔP_n) · Pt5). The flow expands isentropically to
       ambient pressure if that is possible without exceeding Mach 1; otherwise the nozzle
       *chokes* (Mach 1 at the exit) and the exit static pressure stays above ambient.
       Gross thrust (pyCycle's "Cv" convention):
           Fg = Cv · W9 · V9 + A9 (Ps9 − Ps0)
       Net thrust  Fn = Fg − W · V0 (ram drag).   TSFC = W_f / Fn.

Assumptions: steady flow, adiabatic components, no bleed or cooling air (micro
turbojets have uncooled turbines), complete combustion, ideal-gas mixtures with
temperature-dependent cp, fuel mass added in the combustor.

``air_standard=True`` switches to a textbook *air-standard* cycle with a calorically
perfect gas (constant cp, γ) and no fuel mass addition. That mode exists to verify the
code against the closed-form ideal Brayton cycle (tests/test_cycle_simple.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scipy.optimize import brentq

from turboml.gas import thermo
from turboml.physics.atmosphere import isa
from turboml.physics.cycle_common import CycleInputs, CycleResult, Station


class PerfectGas:
    """Calorically perfect gas with the same interface as :class:`thermo.GasMixture`."""

    def __init__(self, cp: float = 1004.5, gamma: float = 1.4):
        self._cp, self._g = cp, gamma
        self.R = cp * (gamma - 1) / gamma

    def cp(self, T):
        return self._cp

    def gamma(self, T):
        return self._g

    def h(self, T):
        return self._cp * T

    def s0(self, T):
        return self._cp * math.log(T)

    def sound_speed(self, T):
        return math.sqrt(self._g * self.R * T)

    def T_from_h(self, h, T_guess=None):
        return h / self._cp

    def T_isentropic(self, T1, p_ratio):
        return T1 * p_ratio ** (self.R / self._cp)

    def p_ratio_isentropic(self, T1, T2):
        return (T2 / T1) ** (self._cp / self.R)


def _nozzle(gas, Tt: float, Pt: float, W: float, Ps0: float):
    """Convergent nozzle exit state. Returns (V, Ts, Ps, A, choked)."""
    ht = gas.h(Tt)

    # Sonic point: ht - h(Ts) = a(Ts)^2 / 2
    def f_sonic(Ts):
        return ht - gas.h(Ts) - 0.5 * gas.sound_speed(Ts) ** 2

    Ts_star = brentq(f_sonic, 0.5 * Tt, Tt, xtol=1e-10)
    Ps_star = Pt * gas.p_ratio_isentropic(Tt, Ts_star)
    if Ps0 < Ps_star:  # choked: exit at Mach 1
        Ts, Ps, choked = Ts_star, Ps_star, True
        V = gas.sound_speed(Ts)
    else:  # subsonic: fully expanded to ambient
        Ps, choked = Ps0, False
        Ts = gas.T_isentropic(Tt, Ps / Pt)
        V = math.sqrt(max(2.0 * (ht - gas.h(Ts)), 0.0))
    rho = Ps / (gas.R * Ts)
    A = W / (rho * V) if V > 0 else float("inf")
    return V, Ts, Ps, A, choked


@dataclass
class FlowState:
    """Total state of the flow at a station: what one component hands to the next.

    ``gas`` is the working-fluid model (air before the combustor, products after it);
    ``h`` is its absolute total enthalpy at ``Tt``.
    """

    Tt: float
    Pt: float
    W: float
    gas: object
    h: float
    far: float = 0.0


@dataclass
class NozzleExit:
    V: float
    Ts: float
    Ps: float
    A: float
    choked: bool
    Fg: float


# ----------------------------------------------------------------------------------------
# Components. Each is a pure function of an incoming FlowState plus its own parameters, so
# Phase 3 can drive them with values from meanline maps (e.g. compressor(pr, eta) at an
# off-design operating point, turbine_given_pr(...)) inside a matching solver.
# ----------------------------------------------------------------------------------------


def free_stream(gas, Ts0: float, Ps0: float, mach: float, W: float) -> tuple[FlowState, float]:
    """Free-stream totals from static state and Mach number. Returns (state, V0)."""
    V0 = mach * gas.sound_speed(Ts0)
    h0 = gas.h(Ts0) + 0.5 * V0**2
    Tt0 = gas.T_from_h(h0, Ts0)
    Pt0 = Ps0 * gas.p_ratio_isentropic(Ts0, Tt0)
    return FlowState(Tt0, Pt0, W, gas, h0), V0


def intake(s: FlowState, recovery: float) -> FlowState:
    """Adiabatic intake: Tt conserved, Pt2 = recovery * Pt0."""
    return replace(s, Pt=recovery * s.Pt)


def compressor(s: FlowState, pr: float, eta: float) -> tuple[FlowState, float]:
    """Compressor with total pressure ratio ``pr`` and isentropic efficiency ``eta``.
    Returns (exit state, shaft power absorbed [W])."""
    g = s.gas
    h_is = g.h(g.T_isentropic(s.Tt, pr))
    h3 = s.h + (h_is - s.h) / eta
    Tt3 = g.T_from_h(h3, s.Tt)
    return FlowState(Tt3, pr * s.Pt, s.W, g, h3, s.far), s.W * (h3 - s.h)


def combustor(
    s: FlowState, tit: float, dp_frac: float, eta_b: float, lhv: float, air_standard_gas=None
) -> FlowState:
    """Burner: find the fuel-air ratio that reaches Tt4 = ``tit`` (absolute-enthalpy energy
    balance, see module docstring). ``air_standard_gas`` switches to the textbook air-standard
    burner (heat addition without fuel mass)."""
    if tit <= s.Tt:
        raise _Infeasible("constraint_tit_below_t3")
    Pt4 = s.Pt * (1.0 - dp_frac)
    if air_standard_gas is not None:
        g = air_standard_gas
        far = (g.h(tit) - s.h) / (eta_b * lhv)
        return FlowState(tit, Pt4, s.W, g, g.h(tit), far)
    h_fuel = thermo.fuel_enthalpy_J_per_kg(lhv, eta_b)

    def resid(f):
        return (1 + f) * thermo.products(f).h(tit) - (s.h + f * h_fuel)

    far = brentq(resid, 0.0, 0.999 * thermo.stoichiometric_far(), xtol=1e-14)
    hot = thermo.products(far)
    return FlowState(tit, Pt4, s.W * (1 + far), hot, hot.h(tit), far)


def turbine_for_power(s: FlowState, power: float, eta: float) -> tuple[FlowState, float]:
    """Turbine delivering shaft ``power`` [W] at isentropic efficiency ``eta`` (design mode).
    Returns (exit state, total pressure ratio Pt4/Pt5)."""
    g = s.gas
    h5 = s.h - power / s.W
    Tt5 = g.T_from_h(h5, s.Tt)
    T5s = g.T_from_h(s.h - (s.h - h5) / eta, Tt5)
    pr = 1.0 / g.p_ratio_isentropic(s.Tt, T5s)
    return FlowState(Tt5, s.Pt / pr, s.W, g, h5, s.far), pr


def turbine_given_pr(s: FlowState, pr: float, eta: float) -> tuple[FlowState, float]:
    """Turbine with imposed total pressure ratio ``pr`` (off-design / map-driven mode).
    Returns (exit state, shaft power produced [W])."""
    g = s.gas
    h_is = g.h(g.T_isentropic(s.Tt, 1.0 / pr))
    h5 = s.h - eta * (s.h - h_is)
    return FlowState(g.T_from_h(h5, s.Tt), s.Pt / pr, s.W, g, h5, s.far), s.W * (s.h - h5)


def convergent_nozzle(s: FlowState, Ps0: float, cv: float, dp_frac: float) -> NozzleExit:
    """Convergent nozzle; gross thrust in pyCycle's "Cv" convention Fg = Cv W V + A (Ps - Ps0)."""
    Pt9 = s.Pt * (1.0 - dp_frac)
    if Pt9 <= Ps0:
        raise _Infeasible("no_nozzle_pressure_ratio")
    V, Ts, Ps, A, choked = _nozzle(s.gas, s.Tt, Pt9, s.W, Ps0)
    return NozzleExit(V, Ts, Ps, A, choked, cv * s.W * V + A * (Ps - Ps0))


class _Infeasible(Exception):
    """Raised inside the cycle for a labelled, physical infeasibility."""


# ----------------------------------------------------------------------------------------


def design_point(x: CycleInputs, air_standard: bool = False, gas=None) -> CycleResult:
    """Run the 0-D design-point cycle.

    Never raises for physics failures. It returns ``converged=False`` with a
    ``failure_reason`` instead (failures are data for the feasibility classifier):
    ``constraint_tit_below_t3``, ``no_nozzle_pressure_ratio`` (turbine leaves less
    than ambient pressure), ``no_net_thrust``, ``no_convergence: ...``.
    """
    try:
        return _design_point(x, air_standard, gas)
    except _Infeasible as err:
        return CycleResult(model="simple", converged=False, failure_reason=str(err))
    except (ValueError, RuntimeError) as err:
        return CycleResult(model="simple", converged=False, failure_reason=f"no_convergence: {err}")


def _design_point(x: CycleInputs, air_standard: bool, gas) -> CycleResult:
    gas = (gas or PerfectGas()) if air_standard else None
    air = gas if air_standard else thermo.air()

    if x.Ts0_K is not None and x.Ps0_Pa is not None:
        Ts0, Ps0 = x.Ts0_K, x.Ps0_Pa
    else:
        amb = isa(x.alt_m, x.dT_isa_K)
        Ts0, Ps0 = amb.T_K, amb.p_Pa

    s0, V0 = free_stream(air, Ts0, Ps0, x.mach, x.W_air_kg_s)
    s2 = intake(s0, x.inlet_recovery)
    s3, P_comp = compressor(s2, x.pr_c, x.eta_c)
    s4 = combustor(s3, x.tit_K, x.dp_burner_frac, x.eta_burner, x.lhv_J_kg, air_standard_gas=gas)
    P_turb = (P_comp + x.power_offtake_W) / x.eta_mech  # shaft balance
    s5, pr_t = turbine_for_power(s4, P_turb, x.eta_t)
    nz = convergent_nozzle(s5, Ps0, x.nozzle_cv, x.dp_nozzle_frac)

    F_ram = x.W_air_kg_s * V0
    Fn = nz.Fg - F_ram
    if Fn <= 0.0:
        raise _Infeasible("no_net_thrust")
    Wf = x.W_air_kg_s * s4.far
    return CycleResult(
        model="simple_airstd" if air_standard else "simple",
        converged=True,
        W_air_kg_s=x.W_air_kg_s,
        Fn_N=Fn,
        Fg_N=nz.Fg,
        F_ram_N=F_ram,
        Wf_kg_s=Wf,
        TSFC_kg_N_s=Wf / Fn,
        far=s4.far,
        pr_t=pr_t,
        P_comp_W=P_comp,
        P_turb_W=P_turb,
        V9_m_s=nz.V,
        Ps9_Pa=nz.Ps,
        A9_m2=nz.A,
        choked=nz.choked,
        stations={
            "0": Station(s0.Tt, s0.Pt),
            "2": Station(s2.Tt, s2.Pt),
            "3": Station(s3.Tt, s3.Pt),
            "4": Station(s4.Tt, s4.Pt),
            "5": Station(s5.Tt, s5.Pt),
            "9": Station(s5.Tt, s5.Pt * (1.0 - x.dp_nozzle_frac)),
        },
    )

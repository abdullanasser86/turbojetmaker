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


def design_point(x: CycleInputs, air_standard: bool = False, gas=None) -> CycleResult:
    """Run the 0-D design-point cycle.

    Never raises for physics failures. It returns ``converged=False`` with a
    ``failure_reason`` instead (failures are data for the feasibility classifier):
    ``constraint_tit_below_t3``, ``no_nozzle_pressure_ratio`` (turbine leaves less
    than ambient pressure), ``no_net_thrust``, ``no_convergence: ...``.
    """
    try:
        return _design_point(x, air_standard, gas)
    except (ValueError, RuntimeError) as err:
        return CycleResult(model="simple", converged=False, failure_reason=f"no_convergence: {err}")


def _design_point(x: CycleInputs, air_standard: bool, gas) -> CycleResult:
    if air_standard:
        gas = gas or PerfectGas()
        air = gas
    else:
        air = thermo.air()

    # --- station 0: free stream
    if x.Ts0_K is not None and x.Ps0_Pa is not None:
        Ts0, Ps0 = x.Ts0_K, x.Ps0_Pa
    else:
        amb = isa(x.alt_m, x.dT_isa_K)
        Ts0, Ps0 = amb.T_K, amb.p_Pa
    V0 = x.mach * air.sound_speed(Ts0)
    h0 = air.h(Ts0) + 0.5 * V0**2
    Tt0 = air.T_from_h(h0, Ts0)
    Pt0 = Ps0 * air.p_ratio_isentropic(Ts0, Tt0)

    # --- 2: intake
    Tt2, Pt2, h2 = Tt0, x.inlet_recovery * Pt0, h0

    # --- 3: compressor
    W = x.W_air_kg_s
    Pt3 = x.pr_c * Pt2
    h3s = air.h(air.T_isentropic(Tt2, x.pr_c))
    h3 = h2 + (h3s - h2) / x.eta_c
    Tt3 = air.T_from_h(h3, Tt2)
    P_comp = W * (h3 - h2)

    # --- 4: combustor
    Pt4 = Pt3 * (1.0 - x.dp_burner_frac)
    if x.tit_K <= Tt3:
        return CycleResult(
            model="simple", converged=False, failure_reason="constraint_tit_below_t3"
        )
    if air_standard:
        far = (gas.h(x.tit_K) - h3) / (x.eta_burner * x.lhv_J_kg)
        hot = gas
        W4 = W  # no fuel mass addition in the air-standard cycle
        h4 = gas.h(x.tit_K)
    else:
        h_fuel = thermo.fuel_enthalpy_J_per_kg(x.lhv_J_kg, x.eta_burner)

        def resid(f):
            return (1 + f) * thermo.products(f).h(x.tit_K) - (h3 + f * h_fuel)

        far = brentq(resid, 0.0, 0.999 * thermo.stoichiometric_far(), xtol=1e-14)
        hot = thermo.products(far)
        W4 = W * (1 + far)
        h4 = hot.h(x.tit_K)
    Wf = W * far

    # --- 5: turbine (shaft power balance)
    P_turb = (P_comp + x.power_offtake_W) / x.eta_mech
    h5 = h4 - P_turb / W4
    Tt5 = hot.T_from_h(h5, x.tit_K)
    h5s = h4 - (h4 - h5) / x.eta_t
    T5s = hot.T_from_h(h5s, Tt5)
    pr_t = 1.0 / hot.p_ratio_isentropic(x.tit_K, T5s)
    Pt5 = Pt4 / pr_t

    # --- 9: convergent nozzle
    Pt9 = Pt5 * (1.0 - x.dp_nozzle_frac)
    if Pt9 <= Ps0:
        return CycleResult(
            model="simple", converged=False, failure_reason="no_nozzle_pressure_ratio"
        )
    V9, _Ts9, Ps9, A9, choked = _nozzle(hot, Tt5, Pt9, W4, Ps0)
    Fg = x.nozzle_cv * W4 * V9 + A9 * (Ps9 - Ps0)
    F_ram = W * V0
    Fn = Fg - F_ram
    if Fn <= 0.0:
        return CycleResult(model="simple", converged=False, failure_reason="no_net_thrust")

    return CycleResult(
        model="simple_airstd" if air_standard else "simple",
        converged=True,
        W_air_kg_s=W,
        Fn_N=Fn,
        Fg_N=Fg,
        F_ram_N=F_ram,
        Wf_kg_s=Wf,
        TSFC_kg_N_s=Wf / Fn,
        far=far,
        pr_t=pr_t,
        P_comp_W=P_comp,
        P_turb_W=P_turb,
        V9_m_s=V9,
        Ps9_Pa=Ps9,
        A9_m2=A9,
        choked=choked,
        stations={
            "0": Station(Tt0, Pt0),
            "2": Station(Tt2, Pt2),
            "3": Station(Tt3, Pt3),
            "4": Station(x.tit_K, Pt4),
            "5": Station(Tt5, Pt5),
            "9": Station(Tt5, Pt9),
        },
    )

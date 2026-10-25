"""Coupled micro-turbojet model: centrifugal compressor + combustor + axial turbine + convergent nozzle
on one shaft (Phase 3).

How an engine "matches"
-----------------------
On a single-spool turbojet the components can't run independently: they share one shaft and one
gas stream. At a given shaft speed N and flight condition, three things must all hold at once:

1. Shaft power balance: the turbine drives the compressor (and any offtake),
       eta_mech * P_turbine = P_compressor + P_offtake.
2. Turbine flow capacity: the turbine, at its inlet state (TIT, p04) and pressure ratio, swallows exactly
   the mass flow the compressor delivers plus the fuel.
3. Nozzle flow capacity: the fixed nozzle area A9 passes that flow at the turbine-exit state.
Unknowns: air mass flow, turbine inlet temperature (i.e. fuel flow) and turbine pressure ratio. The
compressor map (meanline model at (m, N)) supplies PR_c and the compressor power; the turbine model
supplies its mass flow, work and exit state; the nozzle gives the thrust. A damped Newton/Broyden
solver drives the three residuals to zero. Points far from the design are reached by continuation in
speed (steps of 10 % of design speed).

Design mode (`design_engine`): from the cycle design variables (air flow, PR_c, TIT, N) and the
turbine's dimensionless choices, size the compressor (r2 for the target PR, see `size_compressor`),
the turbine (phi, psi, R -> geometry, then annulus/PR correction so it delivers exactly the compressor
power at the design flow) and the nozzle (A9 that passes the design flow). Off-design at the design
speed then reproduces the design point (a built-in consistency check).

Failures are data: every non-converged or infeasible point carries a label, e.g. choke_compressor_*,
turbine_limit_loading, surge_peak_pr, surge_diffuser_stall, constraint_tit_below_t3,
no_nozzle_pressure_ratio, no_convergence. Preliminary design only.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml
from scipy.optimize import brentq

from turboml.gas import thermo
from turboml.physics import cycle_simple as cs
from turboml.physics import mechanical
from turboml.physics.atmosphere import isa
from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry
from turboml.physics.turbine_design import design_stage
from turboml.physics.turbine_meanline import AxialStageGeometry, AxialTurbineMeanline

ROOT = Path(__file__).resolve().parents[2]
ENGINE_CONFIG = ROOT / "configs" / "engine_baseline.yaml"


def _val(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def load_engine_config(path: str | Path = ENGINE_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


class EngineFailure(Exception):
    """Labelled engine-level failure (the label is the message)."""


# ======================================================================================
# Components shared by design and off-design
# ======================================================================================
@dataclass
class ComponentLosses:
    inlet_recovery: float
    dp_burner_frac: float
    eta_burner: float
    eta_mech: float
    dp_nozzle_frac: float
    nozzle_cv: float
    lhv_J_kg: float
    power_offtake_W: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> ComponentLosses:
        c = cfg["components"]
        return cls(**{k: float(_val(c[k])) for k in ("inlet_recovery", "dp_burner_frac", "eta_burner", "eta_mech",
                                                      "dp_nozzle_frac", "nozzle_cv", "lhv_J_kg", "power_offtake_W")})  # fmt: skip


def flight_inlet(alt_m: float, mach: float, dT_isa_K: float, losses: ComponentLosses):
    """Ambient static state, flight speed and compressor-face totals (adiabatic intake)."""
    amb = isa(alt_m, dT_isa_K)
    air = thermo.air()
    fs, V0 = cs.free_stream(air, amb.T_K, amb.p_Pa, mach, 1.0)
    return amb, V0, fs.Tt, fs.Pt * losses.inlet_recovery


def nozzle_mass_flow(
    gas, Tt: float, Pt: float, A9: float, Ps0: float
) -> tuple[float, float, float, bool]:
    """Mass flow a convergent nozzle of throat area A9 passes from total state (Tt, Pt) into Ps0.
    Returns (m, V9, Ps9, choked). Real-gas isentropic expansion (same as cycle_simple._nozzle)."""
    ht = gas.h(Tt)
    Ts_star = brentq(
        lambda Ts: ht - gas.h(Ts) - 0.5 * gas.sound_speed(Ts) ** 2, 0.5 * Tt, Tt, xtol=1e-12
    )
    Ps_star = Pt * gas.p_ratio_isentropic(Tt, Ts_star)
    if Ps0 <= Ps_star:
        Ts, Ps, choked = Ts_star, Ps_star, True
        V = gas.sound_speed(Ts)
    else:
        Ps, choked = Ps0, False
        Ts = gas.T_isentropic(Tt, Ps / Pt)
        V = math.sqrt(max(2.0 * (ht - gas.h(Ts)), 0.0))
    return Ps / (gas.R * Ts) * V * A9, V, Ps, choked


# ======================================================================================
# Compressor sizing (design mode)
# ======================================================================================
def _inducer_velocity(air, mdot, area, T0, p0):
    """Axial velocity C1 with rho(C1) C1 area = mdot (subsonic branch)."""
    h0 = air.h(T0)

    def flux(c):
        T = air.T_from_h(h0 - 0.5 * c * c, T0)
        return p0 * air.p_ratio_isentropic(T0, T) / (air.R * T) * c * area

    c_star = math.sqrt(2 * (h0 - air.h(T0 * 2 / 2.4)))  # ~sonic speed bound
    if flux(c_star) < mdot:
        raise EngineFailure("design_compressor_inducer_choke")
    return brentq(lambda c: flux(c) - mdot, 1e-3, c_star, xtol=1e-10)


@dataclass
class CompressorReference:
    geometry: CentrifugalCompressorGeometry
    delta_sigma: float
    diffuser_cf: float
    phi1t: float  # C1 / U1t at the reference point
    phi2: float  # Cm2 / U2 at the reference point
    incidence_deg: float  # blade LE angle minus mean-radius flow angle at the reference point
    clearance_to_b2: float
    min_clearance_m: float


def compressor_reference(cfg: dict) -> CompressorReference:
    cd = cfg["compressor_design"]
    geom = CentrifugalCompressorGeometry.from_yaml(ROOT / cd["reference_geometry"])
    cal = yaml.safe_load((ROOT / cd["calibration"]).read_text())
    ds, cf = float(_val(cal["delta_sigma"])), float(_val(cal["diffuser_cf"]))
    rp = cd["reference_point"]
    comp = CentrifugalCompressor(geom, delta_sigma=ds, diffuser_cf=cf)
    pt = comp.evaluate(rp["mass_flow_kg_s"], rp["speed_rpm"])
    if not pt.converged:
        raise RuntimeError(f"reference compressor point failed: {pt.failure_reason}")
    omega = rp["speed_rpm"] * math.pi / 30
    c1 = pt.inducer_axial_velocity_m_s
    flow_angle = math.degrees(math.atan(omega * geom.radius_mean_in_m / c1))
    return CompressorReference(geom, ds, cf, c1 / (omega * geom.radius_tip_in_m),
                               pt.exit_meridional_velocity_m_s / (omega * geom.radius_out_m),
                               geom.leading_edge_angle_deg - flow_angle, geom.tip_clearance_m / geom.width_out_m,
                               float(_val(cd["min_tip_clearance_m"])))  # fmt: skip


def size_compressor(ref: CompressorReference, mdot: float, n_rpm: float, pr_target: float, T01: float, p01: float,
                    tol: float = 1e-6, max_iter: int = 40):  # fmt: skip
    """KJ-66-like compressor resized for (mdot, N, PR) -- rules in configs/engine_baseline.yaml.
    Returns (geometry, CompressorPoint at the design point)."""
    air = thermo.air()
    g0 = ref.geometry
    omega = n_rpm * math.pi / 30
    nu = g0.radius_hub_in_m / g0.radius_tip_in_m

    # inducer: same tip flow coefficient C1/U1t as the reference
    def f_ind(r1t):
        area = math.pi * r1t**2 * (1 - nu**2)
        try:
            return _inducer_velocity(air, mdot, area, T01, p01) / (omega * r1t) - ref.phi1t
        except EngineFailure:
            return 1e3  # inducer too small to pass the flow: far above the target flow coefficient

    r1t = brentq(f_ind, 2e-3, 0.3, xtol=1e-9)
    area1 = math.pi * r1t**2 * (1 - nu**2)
    c1 = _inducer_velocity(air, mdot, area1, T01, p01)
    r1m = 0.5 * (1 + nu) * r1t
    le_angle = math.degrees(math.atan(omega * r1m / c1)) + ref.incidence_deg

    # first guess of r2 from the reference: PR ~ f(U2) at the same efficiency
    u2_ref = ref.geometry.radius_out_m * 110000 * math.pi / 30
    cp = air.cp(T01)
    dh_target = cp * T01 * (pr_target ** (air.R / cp) - 1)
    dh_ref = cp * 288.15 * (2.097 ** (air.R / cp) - 1)
    r2 = max(u2_ref * math.sqrt(dh_target / dh_ref) / omega, 1.2 * r1t)
    rho2 = p01 / (air.R * T01) * pr_target ** (1 / 1.4)

    def build(r2, rho2):
        lam = r2 / g0.radius_out_m
        b2 = mdot / (2 * math.pi * r2 * rho2 * ref.phi2 * omega * r2)
        return replace(g0, radius_hub_in_m=nu * r1t, radius_tip_in_m=r1t, radius_out_m=r2, width_out_m=b2,
                       leading_edge_angle_deg=le_angle, length_meridional_m=g0.length_meridional_m * lam,
                       length_axial_m=g0.length_axial_m * lam, diffuser_radius_out_m=g0.diffuser_radius_out_m * lam,
                       diffuser_width_out_m=b2, tip_clearance_m=max(ref.clearance_to_b2 * b2, ref.min_clearance_m),
                       provenance={"sizing": "engine.size_compressor (KJ-66-like rules)"})  # fmt: skip

    # secant on r2 for the target PR; the exit width follows the latest exit density (both converge together)
    pts = []
    for _ in range(max_iter):
        g = build(r2, rho2)
        comp = CentrifugalCompressor(g, delta_sigma=ref.delta_sigma, diffuser_cf=ref.diffuser_cf,
                                     T0_in_K=T01, p0_in_Pa=p01)  # fmt: skip
        pt = comp.evaluate(mdot, n_rpm)
        if not pt.converged:
            raise EngineFailure(f"design_compressor_{pt.failure_reason.split(':')[0]}")
        err = pt.pr_tt / pr_target - 1
        drho = pt.exit_density_kg_m3 / rho2 - 1
        if abs(err) < tol and abs(drho) < 1e-5:  # b2 to 1e-5 relative
            return g, pt
        rho2 = pt.exit_density_kg_m3
        pts.append((r2, err))
        if len(pts) >= 2 and pts[-1][1] != pts[-2][1]:
            (ra, ea), (rb, eb) = pts[-2], pts[-1]
            r_new = rb - eb * (rb - ra) / (eb - ea)
        else:
            r_new = r2 * (1 - 0.5 * err)
        r2 = min(max(r_new, 0.7 * r2), 1.3 * r2)
    raise EngineFailure("design_compressor_pr_not_reached")


# ======================================================================================
# Engine
# ======================================================================================
@dataclass
class EnginePoint:
    n_rpm: float
    alt_m: float
    mach: float
    converged: bool = False
    failure_reason: str = ""
    W_air_kg_s: float = float("nan")
    far: float = float("nan")
    Wf_kg_s: float = float("nan")
    Fn_N: float = float("nan")
    Fg_N: float = float("nan")
    TSFC_kg_N_s: float = float("nan")
    pr_c: float = float("nan")
    eta_c: float = float("nan")
    P_comp_W: float = float("nan")
    tit_K: float = float("nan")
    pr_t_ts: float = float("nan")
    eta_t_ts: float = float("nan")
    P_turb_W: float = float("nan")
    turbine_choked: str = ""
    T02_K: float = float("nan")
    p02_Pa: float = float("nan")
    T03_K: float = float("nan")
    p03_Pa: float = float("nan")
    p04_Pa: float = float("nan")
    T05_K: float = float("nan")  # turbine exit total temperature = EGT
    p05_Pa: float = float("nan")
    V9_m_s: float = float("nan")
    nozzle_choked: bool = False
    surge_flag: str = ""
    compressor_tip_speed_m_s: float = float("nan")
    t0_rel_rotor_K: float = float("nan")
    iterations: int = 0
    max_residual: float = float("nan")
    time_s: float = float("nan")
    tags: list = field(default_factory=list)

    @property
    def specific_thrust_N_s_kg(self) -> float:
        return self.Fn_N / self.W_air_kg_s

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["tags"] = "; ".join(self.tags)
        d["specific_thrust_N_s_kg"] = self.specific_thrust_N_s_kg
        d["TSFC_g_kN_s"] = self.TSFC_kg_N_s * 1e6
        return d


@dataclass
class EngineDesign:
    inputs: dict
    compressor_geometry: CentrifugalCompressorGeometry
    turbine_geometry: AxialStageGeometry
    A9_m2: float
    design_point: EnginePoint
    mechanical: mechanical.MechanicalResult
    turbine_sizing: object = None
    notes: list = field(default_factory=list)


class Engine:
    """A fixed engine geometry, evaluated at off-design conditions (speed-controlled)."""

    def __init__(self, design: EngineDesign, cfg: dict | None = None):
        cfg = cfg or load_engine_config()
        self.cfg, self.design = cfg, design
        self.losses = ComponentLosses.from_config(cfg)
        ref = compressor_reference(cfg) if not hasattr(design, "_ref") else design._ref
        self.comp = CentrifugalCompressor(
            design.compressor_geometry, delta_sigma=ref.delta_sigma, diffuser_cf=ref.diffuser_cf
        )
        self.turb = AxialTurbineMeanline(
            design.turbine_geometry, incidence_sign=cfg["turbine_design"]["incidence_sign"]
        )
        self.A9 = design.A9_m2
        dp = design.design_point
        self.scale = (dp.W_air_kg_s, dp.tit_K, dp.pr_t_ts)
        # design corrected flows W sqrt(Tt)/Pt at intake, burner inlet and nozzle inlet: off-design
        # pressure losses scale with (corrected flow)^2, i.e. with the dynamic head (standard practice,
        # e.g. Walsh & Fletcher, Gas Turbine Performance, 2nd ed., combustor/duct loss scaling --
        # PROVISIONAL, confirm). The design-point fractions are the configured values.
        m4d = dp.W_air_kg_s * (1 + dp.far)
        self.q_design = {"inlet": dp.W_air_kg_s * math.sqrt(dp.T02_K) / (dp.p02_Pa / self.losses.inlet_recovery),
                         "burner": dp.W_air_kg_s * math.sqrt(dp.T03_K) / dp.p03_Pa,
                         "nozzle": m4d * math.sqrt(dp.T05_K) / dp.p05_Pa}  # fmt: skip
        self.loss_scaling = _val(
            cfg["components"].get("off_design_loss_scaling", "corrected_flow_squared")
        )
        self._last = {}  # (alt, mach, dT) -> list of (n_rpm, x)

    # ---------------------------------------------------------------- one evaluation
    def _residuals(self, x, n_rpm, ctx):
        L = self.losses
        m, tit, pr = x[0] * self.scale[0], x[1] * self.scale[1], x[2] * self.scale[2]
        if m <= 0 or pr <= 1.0:
            raise EngineFailure("no_convergence: unphysical iterate")
        scale = self.loss_scaling == "corrected_flow_squared"

        def loss(kind, frac, q):
            return frac * (q / self.q_design[kind]) ** 2 if scale else frac

        # intake: loss from the corrected flow at the free-stream total state (explicit in m)
        p02 = ctx["Pt0"] * (
            1 - loss("inlet", 1 - L.inlet_recovery, m * math.sqrt(ctx["T02"]) / ctx["Pt0"])
        )
        ctx["p02"] = p02
        self.comp.set_inlet(ctx["T02"], p02)
        cp = self.comp.evaluate(m, n_rpm)
        if not cp.converged:
            raise EngineFailure(_comp_label(cp.failure_reason))
        air = thermo.air()
        s3 = cs.FlowState(cp.T03_K, cp.pr_tt * p02, m, air, air.h(cp.T03_K))
        dp_b = loss("burner", L.dp_burner_frac, m * math.sqrt(cp.T03_K) / s3.Pt)
        try:
            s4 = cs.combustor(s3, tit, dp_b, L.eta_burner, L.lhv_J_kg)
        except cs._Infeasible as e:
            raise EngineFailure(str(e)) from e
        except ValueError as e:  # no fuel-air ratio below stoichiometric reaches this TIT
            raise EngineFailure("constraint_tit_above_stoichiometric") from e
        gas = s4.gas
        self.turb.set_gas(gas)
        omega = n_rpm * math.pi / 30
        tp = self.turb.evaluate(tit, s4.Pt, omega, pr)
        if not tp.converged:
            raise EngineFailure(f"turbine_{tp.failure_reason}")
        m4 = s4.W
        Pt9 = tp.p0_out_Pa * (
            1 - loss("nozzle", L.dp_nozzle_frac, m4 * math.sqrt(tp.T0_out_K) / tp.p0_out_Pa)
        )
        if Pt9 <= ctx["Ps0"]:
            raise EngineFailure("no_nozzle_pressure_ratio")
        m9, V9, Ps9, choked9 = nozzle_mass_flow(gas, tp.T0_out_K, Pt9, self.A9, ctx["Ps0"])
        P_c = cp.power_W
        r = [tp.mass_flow_kg_s / m4 - 1.0,
             (L.eta_mech * tp.power_W - P_c - L.power_offtake_W) / max(P_c, 1.0),
             m9 / m4 - 1.0]  # fmt: skip
        state = dict(cp=cp, s4=s4, tp=tp, m4=m4, m9=m9, V9=V9, Ps9=Ps9, choked9=choked9, gas=gas)
        return r, state

    def _point(self, x, state, n_rpm, ctx, it, rmax, t0):
        L = self.losses
        cp, s4, tp = state["cp"], state["s4"], state["tp"]
        m = x[0] * self.scale[0]
        Fg = L.nozzle_cv * state["m4"] * state["V9"] + self.A9 * (state["Ps9"] - ctx["Ps0"])
        Fn = Fg - m * ctx["V0"]
        Wf = m * s4.far
        pt = EnginePoint(n_rpm=n_rpm, alt_m=ctx["alt_m"], mach=ctx["mach"], converged=True, W_air_kg_s=m, far=s4.far,
                         Wf_kg_s=Wf, Fn_N=Fn, Fg_N=Fg, TSFC_kg_N_s=Wf / Fn if Fn > 0 else float("nan"), pr_c=cp.pr_tt,
                         eta_c=cp.eta_tt, P_comp_W=cp.power_W, tit_K=s4.Tt, pr_t_ts=tp.pr_ts, eta_t_ts=tp.eta_ts,
                         P_turb_W=tp.power_W, turbine_choked=tp.choked, T02_K=ctx["T02"], p02_Pa=ctx["p02"],
                         T03_K=cp.T03_K, p03_Pa=cp.pr_tt * ctx["p02"], p04_Pa=s4.Pt, T05_K=tp.T0_out_K,
                         p05_Pa=tp.p0_out_Pa, V9_m_s=state["V9"], nozzle_choked=state["choked9"],
                         compressor_tip_speed_m_s=cp.tip_speed_m_s, t0_rel_rotor_K=tp.T0_rel_rotor_in_K,
                         iterations=it, max_residual=rmax, time_s=time.perf_counter() - t0)  # fmt: skip
        if Fn <= 0:
            pt.converged, pt.failure_reason = False, "no_net_thrust"
        # surge: Senoo diffuser stall, or operating left of the speed line's PR peak (positive slope)
        if cp.diffuser_stall:
            pt.surge_flag = "surge_diffuser_stall"
        else:
            c2 = self.comp.evaluate(m * (1 - 2e-3), n_rpm)
            if c2.converged and c2.pr_tt < cp.pr_tt:
                pt.surge_flag = "surge_peak_pr"
        if pt.surge_flag:
            pt.tags.append(pt.surge_flag)
        return pt

    # ---------------------------------------------------------------- solver
    def _solve(self, n_rpm, ctx, x0, tol=1e-8, max_iter=40):
        """Damped Newton with a finite-difference Jacobian, Broyden updates in between."""
        import numpy as np

        x = np.array(x0, float)
        r, st = self._residuals(x, n_rpm, ctx)
        r = np.array(r)
        J = None
        for it in range(1, max_iter + 1):
            rmax = float(np.max(np.abs(r)))
            if rmax < tol:
                return x, st, it, rmax
            if J is None or it % 6 == 0:
                J = np.empty((3, 3))
                for j in range(3):
                    dx = np.zeros(3)
                    dx[j] = 1e-5 * max(abs(x[j]), 0.1)
                    rj, _ = self._residuals(x + dx, n_rpm, ctx)
                    J[:, j] = (np.array(rj) - r) / dx[j]
            step = -np.linalg.solve(J, r)
            lim = np.max(np.abs(step) / np.array([0.08, 0.08, 0.15]))
            if lim > 1:
                step /= lim
            lam = 1.0
            for _ in range(12):
                try:
                    r_new, st_new = self._residuals(x + lam * step, n_rpm, ctx)
                    r_new = np.array(r_new)
                    if np.max(np.abs(r_new)) < (1 - 1e-4 * lam) * rmax or lam < 0.05:
                        break
                except EngineFailure:
                    pass
                lam *= 0.5
            else:
                raise EngineFailure("no_convergence: line search failed")
            dxs = lam * step
            dr = r_new - r
            J = J + np.outer(dr - J @ dxs, dxs) / float(dxs @ dxs)  # Broyden update
            x, r, st = x + dxs, r_new, st_new
        raise EngineFailure("no_convergence: max iterations")

    def off_design(self, n_rpm: float, alt_m: float = 0.0, mach: float = 0.0, dT_isa_K: float = 0.0,
                   x0=None, continuation_step: float = 0.1) -> EnginePoint:  # fmt: skip
        """Operating point at shaft speed n_rpm and flight condition. Never raises (failures are data)."""
        t0 = time.perf_counter()
        amb, V0, T02, p02 = flight_inlet(alt_m, mach, dT_isa_K, self.losses)
        ctx = dict(
            alt_m=alt_m,
            mach=mach,
            Ps0=amb.p_Pa,
            V0=V0,
            T02=T02,
            p02=p02,
            Pt0=p02 / self.losses.inlet_recovery,
        )
        dp = self.design.design_point
        key = (alt_m, mach, dT_isa_K)
        try:
            if x0 is None:
                # continuation in speed from the nearest solved point (or the design point)
                hist = self._last.get(key) or [(dp.n_rpm, None)]
                n_prev, x_prev = min(hist, key=lambda h: abs(h[0] - n_rpm))
                if x_prev is None:
                    # design-point solution, corrected for the inlet state (corrected flow)
                    x_prev = [p02 / dp.p02_Pa * math.sqrt(dp.T02_K / T02), 1.0, 1.0]
                    if abs(p02 / dp.p02_Pa - 1) > 0.05 or abs(T02 / dp.T02_K - 1) > 0.02:
                        x_prev, n_prev = (
                            self._continue_inlet(n_rpm=dp.n_rpm, ctx=ctx, key=key),
                            dp.n_rpm,
                        )
                speeds = _speed_steps(n_prev, n_rpm, continuation_step * dp.n_rpm)
                x = x_prev
                for n in speeds:
                    guess = _extrapolate(self._last.get(key, []), n, x)
                    x, st, it, rmax = self._solve(n, ctx, guess)
                    self._last.setdefault(key, []).append((n, list(x)))
            else:
                x, st, it, rmax = self._solve(n_rpm, ctx, x0)
            pt = self._point(x, st, n_rpm, ctx, it, rmax, t0)
        except EngineFailure as e:
            pt = EnginePoint(
                n_rpm=n_rpm,
                alt_m=alt_m,
                mach=mach,
                failure_reason=str(e),
                time_s=time.perf_counter() - t0,
            )
        return pt

    def _continue_inlet(self, n_rpm, ctx, key, steps=4):
        """Continuation from SLS design to a different inlet state (altitude / flight Mach) at design speed."""
        dp = self.design.design_point
        x = [1.0, 1.0, 1.0]
        for k in range(1, steps + 1):
            f = k / steps
            c = dict(ctx)
            for name, d0 in (("T02", dp.T02_K), ("Pt0", dp.p02_Pa / self.losses.inlet_recovery),
                             ("Ps0", self.design.inputs["Ps0_Pa"]), ("V0", 0.0)):  # fmt: skip
                c[name] = d0 + f * (ctx[name] - d0)
            x, _, _, _ = self._solve(n_rpm, c, x)
        self._last.setdefault(key, []).append((n_rpm, list(x)))
        return list(x)


def _speed_steps(n0, n1, step):
    k = max(1, math.ceil(abs(n1 - n0) / step - 1e-9))
    return [n0 + (n1 - n0) * (i + 1) / k for i in range(k)]


def _extrapolate(hist, n, x):
    """Linear extrapolation of the solution vector in speed from the last two solved points."""
    pts = sorted(hist, key=lambda h: abs(h[0] - n))[:2]
    if len(pts) < 2 or pts[0][0] == pts[1][0]:
        return x
    (na, xa), (nb, xb) = pts
    return [xa[i] + (xb[i] - xa[i]) * (n - na) / (nb - na) for i in range(3)]


def _comp_label(reason: str) -> str:
    r = reason.split(":")[0]
    if r.startswith("choke"):
        return f"choke_compressor_{r.removeprefix('choke_')}"
    return f"compressor_{r}"


# ======================================================================================
# Design mode
# ======================================================================================
def design_engine(inputs: dict, cfg: dict | None = None) -> EngineDesign:
    """Size an engine for the design-point inputs (keys as in configs/engine_baseline.yaml
    `baseline_design`). Raises EngineFailure with a label if the design can't be built."""
    cfg = cfg or load_engine_config()
    L = ComponentLosses.from_config(cfg)
    x = {k: float(_val(v)) for k, v in inputs.items()}
    amb, V0, T02, p02 = flight_inlet(
        x.get("alt_m", 0.0), x.get("mach", 0.0), x.get("dT_isa_K", 0.0), L
    )
    ref = compressor_reference(cfg)
    m, N, tit = x["W_air_kg_s"], x["n_rpm"], x["tit_K"]
    cg, cpt = size_compressor(ref, m, N, x["pr_c"], T02, p02)
    air = thermo.air()
    s3 = cs.FlowState(cpt.T03_K, cpt.pr_tt * p02, m, air, air.h(cpt.T03_K))
    try:
        s4 = cs.combustor(s3, tit, L.dp_burner_frac, L.eta_burner, L.lhv_J_kg)
    except cs._Infeasible as e:
        raise EngineFailure(str(e)) from e
    work = (cpt.power_W + L.power_offtake_W) / L.eta_mech / s4.W
    omega = N * math.pi / 30
    try:
        sz, turb, tpt = design_stage(s4.W, tit, s4.Pt, omega, work, x["turb_phi"], x["turb_psi"], x["turb_reaction"],
                                     gas=s4.gas, incidence_sign=cfg["turbine_design"]["incidence_sign"])  # fmt: skip
    except ValueError as e:
        raise EngineFailure(f"design_turbine: {e}") from e
    Pt9 = tpt.p0_out_Pa * (1 - L.dp_nozzle_frac)
    if Pt9 <= amb.p_Pa:
        raise EngineFailure("no_nozzle_pressure_ratio")
    noz = cs.convergent_nozzle(cs.FlowState(tpt.T0_out_K, tpt.p0_out_Pa, s4.W, s4.gas, s4.gas.h(tpt.T0_out_K), s4.far),
                               amb.p_Pa, L.nozzle_cv, L.dp_nozzle_frac)  # fmt: skip
    Fn = noz.Fg - m * V0
    Wf = m * s4.far
    dp = EnginePoint(n_rpm=N, alt_m=x.get("alt_m", 0.0), mach=x.get("mach", 0.0), converged=Fn > 0,
                     failure_reason="" if Fn > 0 else "no_net_thrust", W_air_kg_s=m, far=s4.far, Wf_kg_s=Wf, Fn_N=Fn,
                     Fg_N=noz.Fg, TSFC_kg_N_s=Wf / Fn if Fn > 0 else float("nan"), pr_c=cpt.pr_tt, eta_c=cpt.eta_tt,
                     P_comp_W=cpt.power_W, tit_K=tit, pr_t_ts=tpt.pr_ts, eta_t_ts=tpt.eta_ts, P_turb_W=tpt.power_W,
                     turbine_choked=tpt.choked, T02_K=T02, p02_Pa=p02, T03_K=cpt.T03_K, p03_Pa=s3.Pt, p04_Pa=s4.Pt,
                     T05_K=tpt.T0_out_K, p05_Pa=tpt.p0_out_Pa, V9_m_s=noz.V, nozzle_choked=noz.choked,
                     compressor_tip_speed_m_s=cpt.tip_speed_m_s, t0_rel_rotor_K=tpt.T0_rel_rotor_in_K)  # fmt: skip
    mats = mechanical.MaterialSet.from_yaml()
    from turboml.config import load_config

    tip_limit = load_config().constraints.max_impeller_tip_speed_m_s
    mech = mechanical.evaluate(tit, omega, cg.radius_out_m, sz.geometry.rotor_annulus_area_m2, tip_limit, mats,
                               t0_rel_rotor_K=tpt.T0_rel_rotor_in_K)  # fmt: skip
    dp.tags.extend(mech.tags)
    design = EngineDesign(inputs={**x, "Ps0_Pa": amb.p_Pa}, compressor_geometry=cg, turbine_geometry=sz.geometry,
                          A9_m2=noz.A, design_point=dp, mechanical=mech, turbine_sizing=sz,
                          notes=list(sz.warnings))  # fmt: skip
    design._ref = ref
    return design


# ======================================================================================
# Design card
# ======================================================================================
SAFETY_NOTE = ("PRELIMINARY DESIGN ONLY - NOT MANUFACTURING-READY. Real micro turbines spin at 100 000+ rpm and "
               "need detailed 3-D aerodynamic, structural, rotordynamic, thermal and combustion analysis plus "
               "physical testing by qualified people before anything is built or run.")  # fmt: skip


def design_card(
    d: EngineDesign,
    off_design: list[tuple[str, EnginePoint]] | None = None,
    title: str = "Engine design",
) -> str:
    p, cg, tg, mech = d.design_point, d.compressor_geometry, d.turbine_geometry, d.mechanical
    env = mech.checks["envelope"]
    lines = [f"# {title}", "", f"> **{SAFETY_NOTE}**", "",
             "Status: physics-evaluated design point (0-D cycle + meanline components); not surrogate output.", ""]  # fmt: skip
    if mech.premium_material_required:
        lines += [
            "**Tag: premium material required** (turbine temperature above the default material's limit).",
            "",
        ]
    lines += ["## Design point", "", "| quantity | value |", "|---|---|",
              f"| net thrust | {p.Fn_N:.1f} N |", f"| air mass flow | {p.W_air_kg_s:.3f} kg/s |",
              f"| specific thrust | {p.specific_thrust_N_s_kg:.0f} N s/kg |",
              f"| TSFC | {p.TSFC_kg_N_s * 1e6:.1f} g/(kN s) ({p.TSFC_kg_N_s * 3600:.3f} kg/(N h)) |",
              f"| fuel flow | {p.Wf_kg_s * 1000:.2f} g/s |", f"| shaft speed | {p.n_rpm:.0f} rpm |",
              f"| compressor PR / eta_tt | {p.pr_c:.3f} / {p.eta_c:.3f} |",
              f"| TIT | {p.tit_K:.0f} K |", f"| turbine PR_ts / eta_ts | {p.pr_t_ts:.3f} / {p.eta_t_ts:.3f} |",
              f"| EGT (turbine exit total) | {p.T05_K:.0f} K ({p.T05_K - 273.15:.0f} C) |",
              f"| nozzle area A9 | {d.A9_m2 * 1e4:.2f} cm2 ({'choked' if p.nozzle_choked else 'subsonic'}) |", ""]  # fmt: skip
    lines += ["## Geometry (meanline)", "", "| part | value |", "|---|---|",
              f"| impeller tip radius r2 | {cg.radius_out_m * 1e3:.2f} mm (tip speed {p.compressor_tip_speed_m_s:.0f} m/s) |",
              f"| inducer hub / tip radius | {cg.radius_hub_in_m * 1e3:.2f} / {cg.radius_tip_in_m * 1e3:.2f} mm |",
              f"| impeller exit width b2 | {cg.width_out_m * 1e3:.2f} mm |",
              f"| blades / backsweep | {cg.number_of_blades} / {cg.trailing_edge_angle_deg:.0f} deg |",
              f"| diffuser outlet radius | {cg.diffuser_radius_out_m * 1e3:.1f} mm (vaneless model) |",
              f"| turbine mean radius | {(tg.radius_tip_out[1] + tg.radius_hub_out[1]) / 2 * 1e3:.2f} mm |",
              f"| turbine rotor tip radius / blade height | {tg.rotor_tip_radius_m * 1e3:.2f} / {(tg.radius_tip_out[1] - tg.radius_hub_out[1]) * 1e3:.2f} mm |",
              f"| turbine blades (stator / rotor, non-integer sizing) | {d.turbine_sizing.n_blades[0]:.1f} / {d.turbine_sizing.n_blades[1]:.1f} |",
              f"| envelope D x L (proxy) | {env['diameter_m'] * 1e3:.0f} x {env['length_m'] * 1e3:.0f} mm |",
              f"| dry mass (proxy) | {env['dry_mass_kg']:.2f} +- {env['dry_mass_1sigma_kg']:.2f} kg |", ""]  # fmt: skip
    t = mech.checks["turbine_temperature"]
    s = mech.checks["blade_root_stress"]
    lines += ["## Constraints and proxies", "",
              f"- material: {t['material'] or 'NONE (infeasible)'}; metal temperature basis {t['basis']}: {t['metal_temperature_K']:.0f} K; "
              f"margins {', '.join(f'{k} {v:+.0f} K' for k, v in t['margins_K'].items())}",
              f"- rotor inlet relative total temperature: {p.t0_rel_rotor_K:.0f} K",
              f"- impeller tip speed {mech.checks['impeller_tip_speed']['value_m_s']:.0f} m/s (limit {mech.checks['impeller_tip_speed']['limit_m_s']:.0f})",
              f"- blade root stress proxy: {s.get('root_stress_Pa', float('nan')) / 1e6:.0f} MPa ({s['status']}); AN^2 = {s['an2_m2_rpm2']:.3g} m2 rpm2",
              f"- feasible: {mech.feasible}" + (f" ({', '.join(mech.infeasible_reasons)})" if mech.infeasible_reasons else ""),
              f"- tags: {', '.join(p.tags) or 'none'}", ""]  # fmt: skip
    if d.notes:
        lines += ["Sizing warnings: " + "; ".join(d.notes), ""]
    if off_design:
        lines += ["## Off-design (physics-evaluated, speed-controlled)", "",
                  "| point | N [rpm] | alt [m] | Mach | Fn [N] | TSFC [g/(kN s)] | m_air [kg/s] | PR_c | TIT [K] | EGT [K] | status |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for name, q in off_design:
            status = "ok" if q.converged else q.failure_reason
            if q.surge_flag:
                status += f" ({q.surge_flag})"
            lines.append(f"| {name} | {q.n_rpm:.0f} | {q.alt_m:.0f} | {q.mach:.2f} | {q.Fn_N:.1f} | {q.TSFC_kg_N_s * 1e6:.1f} | "
                         f"{q.W_air_kg_s:.3f} | {q.pr_c:.3f} | {q.tit_K:.0f} | {q.T05_K:.0f} | {status} |")  # fmt: skip
        lines.append("")
    lines += ["Component assumptions: configs/engine_baseline.yaml (DePaola 2020 values as summarised by the user; "
              "compressor calibration from the KJ-66, provisional).", ""]  # fmt: skip
    return "\n".join(lines)


def copy_design(d: EngineDesign) -> EngineDesign:
    return copy.deepcopy(d)

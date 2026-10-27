"""Sizing a single-stage axial turbine from dimensionless design choices (design mode).

What this does
--------------
In preliminary design you don't start from a blade drawing; you choose a few dimensionless numbers
that say what kind of stage you want and derive the geometry from them:

* loading coefficient   psi = dh0 / U^2   (how much work per unit blade speed squared),
* flow coefficient      phi = C_x / U     (how fast the gas moves axially relative to the blades),
* reaction              R                 (share of the static-enthalpy drop taken in the rotor).

With the required specific work dh0 (it must drive the compressor) and the shaft speed omega:
    U = sqrt(dh0 / psi),  r_mean = U / omega,  C_x = phi U,
and the velocity triangles (constant C_x and radius; repeating-stage relations, Dixon & Hall Ch. 4):
    C_theta2 = U (1 - R + psi/2),  C_theta3 = U (1 - R - psi/2),
    alpha2 = atan(C_theta2/C_x), beta2 = atan((C_theta2 - U)/C_x), beta3 = atan((C_theta3 - U)/C_x).
Blade heights come from continuity at each station, A = m / (rho C_x) = 2 pi r_mean h, with densities
from first-guess efficiencies (config); `design_stage` then corrects the annulus size and finds the
pressure ratio with the full meanline model so that the stage passes exactly the design mass flow
while delivering exactly the design work.
Blade rows: metal inlet angle = flow inlet angle (zero incidence); the throat ("gauging") angle is
chosen so that the Aungier deviation model returns the wanted exit flow angle at the design Mach
number; pitch from Zweifel's loading coefficient; chord from the aspect ratio; maximum thickness from
TurboFlow's camber rule; trailing edge, leading edge and tip clearance from ratios in
configs/components/axial_turbine_design.yaml. All of this is preliminary design only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from scipy.optimize import brentq

from turboml.gas.thermo import GasMixture, air
from turboml.physics.turbine_meanline import AxialStageGeometry, AxialTurbineMeanline

ROOT = Path(__file__).resolve().parents[2]
DESIGN_DEFAULTS = ROOT / "configs" / "components" / "axial_turbine_design.yaml"
_D2R = math.pi / 180.0

# TurboFlow check_turbine_geometry recommended ranges (geometry_model.py) -> warnings, not errors
_RECOMMENDED = {
    "hub_tip_ratio": (0.50, 0.95), "aspect_ratio": (0.8, 5.0), "pitch_chord_ratio": (0.3, 1.1),
    "stagger_angle": (-10.0, 70.0), "leading_edge_angle": (-60.0, 25.0),
}  # fmt: skip


def load_design_defaults(path: str | Path = DESIGN_DEFAULTS) -> dict:
    return {
        k: v["value"] for k, v in yaml.safe_load(Path(path).read_text(encoding="utf-8")).items()
    }


@dataclass
class StageSizing:
    geometry: AxialStageGeometry
    U_m_s: float
    r_mean_m: float
    C_x_m_s: float
    angles_deg: dict
    mach: dict
    heights_m: list
    n_blades: list
    pr_ts_estimate: float
    warnings: list = field(default_factory=list)


def _delta0(g):
    beta_g = 90.0 - g
    cg = math.cos(g * _D2R)
    return math.degrees(math.asin(min(1.0, cg * (1 + (1 - cg) * (beta_g / 90.0) ** 2)))) - beta_g


def _gauging_for(target_deg: float, mach: float, mach_crit: float) -> float:
    """Gauging angle g [deg, unsigned] whose Aungier exit flow angle at `mach` equals |target|."""
    t = abs(target_deg)
    X = (mach - 0.5) / (mach_crit - 0.5)
    p = 1.0 if X < 0 else (0.0 if X > 1 else 1 - 10 * X**3 + 15 * X**4 - 6 * X**5)
    if p == 0.0:
        return t
    return brentq(lambda g: g - _delta0(g) * p - t, t, min(t + 15.0, 89.0))


def _thickness_rule(camber_deg: float, chord: float) -> float:
    """TurboFlow prepare_geometry: t_max/c = 0.15 (camber <= 40), +1.25e-3/deg up to 120, 0.25 above."""
    if camber_deg <= 40:
        return 0.15 * chord
    if camber_deg <= 120:
        return (0.15 + 1.25e-3 * (camber_deg - 40)) * chord
    return 0.25 * chord


def size_axial_stage(mdot_kg_s: float, T0_in_K: float, p0_in_Pa: float, omega_rad_s: float, work_J_kg: float,
                     phi: float, psi: float, reaction: float, gas: GasMixture | None = None,
                     height_scale: float = 1.0, **overrides) -> StageSizing:  # fmt: skip
    """Geometry of a single axial stage from the design point and dimensionless choices (see module doc)."""
    gas = gas or air()
    d = {**load_design_defaults(), **overrides}
    U = math.sqrt(work_J_kg / psi)
    r_m = U / omega_rad_s
    Cx = phi * U
    a1 = d["stator_inlet_angle_deg"]
    Ct1 = Cx * math.tan(a1 * _D2R)
    Ct2 = U * (1 - reaction + psi / 2)
    Ct3 = U * (1 - reaction - psi / 2)
    a2 = math.degrees(math.atan(Ct2 / Cx))
    b2 = math.degrees(math.atan((Ct2 - U) / Cx))
    b3 = math.degrees(math.atan((Ct3 - U) / Cx))
    a3 = math.degrees(math.atan(Ct3 / Cx))

    # station states (first-guess efficiencies; corrected by design_stage)
    h0 = gas.h(T0_in_K)

    def static(h0x, c2, T_guess):
        return gas.T_from_h(h0x - 0.5 * c2, T_guess)

    T1 = static(h0, Cx**2 + Ct1**2, T0_in_K)
    p1 = p0_in_Pa * gas.p_ratio_isentropic(T0_in_K, T1)
    T2 = static(h0, Cx**2 + Ct2**2, T0_in_K)
    h2s = h0 - (h0 - gas.h(T2)) / d["eta_stator_estimate"]
    p2 = p0_in_Pa * gas.p_ratio_isentropic(T0_in_K, gas.T_from_h(h2s, T2))
    h03 = h0 - work_J_kg
    T03 = gas.T_from_h(h03, T0_in_K)
    T03s = gas.T_from_h(h0 - work_J_kg / d["eta_stage_tt_estimate"], T03)
    p03 = p0_in_Pa * gas.p_ratio_isentropic(T0_in_K, T03s)
    T3 = static(h03, Cx**2 + Ct3**2, T03)
    p3 = p03 * gas.p_ratio_isentropic(T03, T3)
    rho = [p / (gas.R * T) for p, T in ((p1, T1), (p2, T2), (p3, T3))]
    heights = [height_scale * mdot_kg_s / (r * Cx) / (2 * math.pi * r_m) for r in rho]
    if heights[2] / 2 >= r_m:
        raise ValueError(
            "blade height exceeds the mean diameter (hub radius <= 0): flow coefficient too low"
        )
    a_snd = [math.sqrt(gas.gamma(T) * gas.R * T) for T in (T1, T2, T3)]
    M2 = math.hypot(Cx, Ct2) / a_snd[1]
    Mw3 = math.hypot(Cx, Ct3 - U) / a_snd[2]
    mc = d["critical_mach_estimate"]
    g_st = _gauging_for(a2, M2, mc)
    g_ro = _gauging_for(b3, Mw3, mc)

    # rows: [stator, rotor]; stations stator 1->2, rotor 2->3
    h_in, h_out = [heights[0], heights[1]], [heights[1], heights[2]]
    le = [a1, b2]
    gauging_signed = [g_st, -g_ro]
    flow_in, flow_out = [a1, b2], [a2, b3]
    geom: dict = {k: [0.0, 0.0] for k in (
        "radius_hub_in", "radius_hub_out", "radius_tip_in", "radius_tip_out", "pitch", "chord",
        "stagger_angle", "opening", "leading_edge_angle", "leading_edge_wedge_angle",
        "leading_edge_diameter", "trailing_edge_thickness", "maximum_thickness", "tip_clearance",
        "throat_location_fraction")}  # fmt: skip
    n_blades = []
    fr = d["throat_location_fraction"]
    for i in range(2):
        geom["radius_hub_in"][i], geom["radius_tip_in"][i] = r_m - h_in[i] / 2, r_m + h_in[i] / 2
        geom["radius_hub_out"][i], geom["radius_tip_out"][i] = (
            r_m - h_out[i] / 2,
            r_m + h_out[i] / 2,
        )
        h_mean = 0.5 * (h_in[i] + h_out[i])
        chord = h_mean / d["aspect_ratio"]
        stagger = 0.5 * (le[i] + gauging_signed[i])
        cx = chord * math.cos(stagger * _D2R)
        ta, tb = math.tan(flow_in[i] * _D2R), math.tan(flow_out[i] * _D2R)
        pitch = (
            d["zweifel_coefficient"] * cx / (2 * math.cos(flow_out[i] * _D2R) ** 2 * abs(ta - tb))
        )
        h_th = (1 - fr) * h_in[i] + fr * h_out[i]
        A_out = 2 * math.pi * r_m * h_out[i]
        opening = A_out * math.cos(gauging_signed[i] * _D2R) * pitch / (2 * math.pi * r_m * h_th)
        geom["pitch"][i], geom["chord"][i], geom["stagger_angle"][i], geom["opening"][i] = (
            pitch,
            chord,
            stagger,
            opening,
        )
        geom["leading_edge_angle"][i] = le[i]
        geom["leading_edge_wedge_angle"][i] = d["leading_edge_wedge_angle_deg"]
        geom["leading_edge_diameter"][i] = d["leading_edge_diameter_chord_ratio"] * chord
        geom["trailing_edge_thickness"][i] = max(d["trailing_edge_thickness_opening_ratio"] * opening,
                                                 d["min_trailing_edge_thickness_m"])  # fmt: skip
        geom["maximum_thickness"][i] = _thickness_rule(abs(le[i] - gauging_signed[i]), chord)
        geom["tip_clearance"][i] = (
            0.0
            if i == 0
            else max(d["tip_clearance_height_ratio"] * h_mean, d["min_tip_clearance_m"])
        )
        geom["throat_location_fraction"][i] = fr
        n_blades.append(2 * math.pi * r_m / pitch)
    g = AxialStageGeometry(**geom)

    warnings = []
    for i, row in enumerate(("stator", "rotor")):
        checks = {
            "hub_tip_ratio": min(geom["radius_hub_in"][i] / geom["radius_tip_in"][i],
                                 geom["radius_hub_out"][i] / geom["radius_tip_out"][i]),
            "aspect_ratio": 0.5 * (h_in[i] + h_out[i]) / geom["chord"][i],
            "pitch_chord_ratio": geom["pitch"][i] / geom["chord"][i],
            "stagger_angle": abs(geom["stagger_angle"][i]),
            "leading_edge_angle": le[i] if i == 0 else -le[i],
        }  # fmt: skip
        for k, v in checks.items():
            lo, hi = _RECOMMENDED[k]
            if not lo <= v <= hi:
                warnings.append(f"{row} {k} = {v:.3g} outside TurboFlow's recommended [{lo}, {hi}]")
    return StageSizing(
        geometry=g, U_m_s=U, r_mean_m=r_m, C_x_m_s=Cx,
        angles_deg={"alpha1": a1, "alpha2": a2, "beta2": b2, "beta3": b3, "alpha3": a3,
                    "gauging_stator": g_st, "gauging_rotor": -g_ro},
        mach={"stator_exit": M2, "rotor_exit_rel": Mw3}, heights_m=heights, n_blades=n_blades,
        pr_ts_estimate=p0_in_Pa / p3, warnings=warnings,
    )  # fmt: skip


def design_stage(mdot_kg_s: float, T0_in_K: float, p0_in_Pa: float, omega_rad_s: float, work_J_kg: float,
                 phi: float, psi: float, reaction: float, gas: GasMixture | None = None,
                 incidence_sign: str = "turboflow", tol: float = 1e-6, max_iter: int = 30, **overrides):  # fmt: skip
    """Size the stage, then adjust the annulus (blade-height scale k) and the total-to-static pressure
    ratio with the meanline model until it passes `mdot_kg_s` and delivers `work_J_kg` exactly.

    Two unknowns (k, PR_ts), two residuals (mass flow, work); Newton with finite-difference Jacobian.
    Returns (StageSizing, AxialTurbineMeanline, TurbinePoint at the design point). Raises ValueError
    with a reason if the stage can't deliver the work (e.g. limit loading)."""
    gas = gas or air()
    k = 1.0
    sz = size_axial_stage(
        mdot_kg_s,
        T0_in_K,
        p0_in_Pa,
        omega_rad_s,
        work_J_kg,
        phi,
        psi,
        reaction,
        gas,
        k,
        **overrides,
    )
    pr = sz.pr_ts_estimate

    def evaluate(k, pr):
        s = size_axial_stage(
            mdot_kg_s,
            T0_in_K,
            p0_in_Pa,
            omega_rad_s,
            work_J_kg,
            phi,
            psi,
            reaction,
            gas,
            k,
            **overrides,
        )
        t = AxialTurbineMeanline(s.geometry, gas, incidence_sign=incidence_sign)
        pt = t.evaluate(T0_in_K, p0_in_Pa, omega_rad_s, pr)
        if not pt.converged:
            raise ValueError(f"turbine design point failed: {pt.failure_reason} ({pt.message})")
        return s, t, pt, (pt.mass_flow_kg_s / mdot_kg_s - 1.0, pt.work_J_kg / work_J_kg - 1.0)

    for _ in range(max_iter):
        s, t, pt, r = evaluate(k, pr)
        if max(abs(r[0]), abs(r[1])) < tol:
            return s, t, pt
        # Jacobian in (ln k, ln pr)
        dk, dp = 1e-4, 1e-4
        _, _, _, rk = evaluate(k * (1 + dk), pr)
        _, _, _, rp = evaluate(k, pr * (1 + dp))
        J = [[(rk[0] - r[0]) / dk, (rp[0] - r[0]) / dp], [(rk[1] - r[1]) / dk, (rp[1] - r[1]) / dp]]
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        if abs(det) < 1e-14:
            raise ValueError(
                "turbine design solve: singular Jacobian (choked stage cannot change its work?)"
            )
        x0 = -(J[1][1] * r[0] - J[0][1] * r[1]) / det
        x1 = -(-J[1][0] * r[0] + J[0][0] * r[1]) / det
        x0, x1 = max(-0.3, min(0.3, x0)), max(-0.3, min(0.3, x1))
        k *= math.exp(x0)
        pr = max(1.05, pr * math.exp(x1))
    raise ValueError("turbine design solve did not converge")


def rotor_exit_hub_tip(geometry: AxialStageGeometry) -> float:
    """Rotor-exit hub-to-tip radius ratio (the station with the tallest blades of the stage)."""
    return geometry.radius_hub_out[1] / geometry.radius_tip_out[1]


def design_stage_hub_tip(mdot_kg_s: float, T0_in_K: float, p0_in_Pa: float, omega_rad_s: float, work_J_kg: float,
                         hub_tip: float, psi: float, reaction: float, gas: GasMixture | None = None,
                         incidence_sign: str = "turboflow", tol: float = 1e-6, max_iter: int = 30, **overrides):  # fmt: skip
    """`design_stage` with the rotor-exit hub/tip ratio prescribed instead of the flow coefficient phi.

    With psi, the work and the speed fixed, the mean radius r_m = sqrt(work/psi)/omega is fixed too, so the
    hub/tip ratio lam fixes the rotor-exit blade height h = 2 r_m (1 - lam)/(1 + lam). Continuity then sets
    the axial velocity, i.e. phi = C_x/U (blade height ~ 1/phi at fixed mass flow). We iterate
    phi <- phi * h_actual/h_target on the converged `design_stage` geometry (fixed point, ~3-5 passes).

    Returns (StageSizing, AxialTurbineMeanline, TurbinePoint, phi). Raises ValueError with a reason.
    """
    if not 0.0 < hub_tip < 1.0:
        raise ValueError(f"hub/tip ratio {hub_tip} outside (0, 1)")
    gas = gas or air()
    U = math.sqrt(work_J_kg / psi)
    r_m = U / omega_rad_s
    h_target = 2 * r_m * (1 - hub_tip) / (1 + hub_tip)
    # first guess of phi from continuity at the rotor exit (static density ~ total density at the exit)
    T03 = gas.T_from_h(gas.h(T0_in_K) - work_J_kg, T0_in_K)
    rho3 = p0_in_Pa * gas.p_ratio_isentropic(T0_in_K, T03) / (gas.R * T03)
    phi = mdot_kg_s / (rho3 * 2 * math.pi * r_m * h_target) / U
    a03 = math.sqrt(
        gas.gamma(T03) * gas.R * T03
    )  # an axial velocity above this would be supersonic

    def check(phi):
        if phi * U >= a03:
            raise ValueError(
                f"rotor hub/tip {hub_tip:.3f} too high for the flow: needs axial Mach >= 1 (phi ~ {phi:.3g})"
            )
        if phi <= 0.05:
            raise ValueError(f"rotor hub/tip {hub_tip:.3f} too low for the flow (phi ~ {phi:.3g})")

    for _ in range(max_iter):
        check(phi)
        sz, t, pt = design_stage(mdot_kg_s, T0_in_K, p0_in_Pa, omega_rad_s, work_J_kg, phi, psi, reaction, gas,
                                 incidence_sign, tol, **overrides)  # fmt: skip
        g = sz.geometry
        h = g.radius_tip_out[1] - g.radius_hub_out[1]
        if abs(h / h_target - 1) < 1e-5:
            return sz, t, pt, phi
        phi *= h / h_target
    raise ValueError("hub/tip sizing did not converge")


def max_hub_tip_for_exit_mach(mdot_kg_s: float, T0_in_K: float, p0_in_Pa: float, omega_rad_s: float,
                              work_J_kg: float, psi: float, reaction: float, gas: GasMixture | None = None,
                              mach_max: float = 0.5, **overrides) -> float | None:  # fmt: skip
    """Largest rotor-exit hub/tip ratio whose absolute exit Mach number stays at or below `mach_max`
    (Phase 4 option T1; DePaola 2020 Table 2: turbine exit Mach 0.4-0.5).

    Derivation (same first-guess states as `size_axial_stage`):
      U = sqrt(work/psi), r_m = U/omega, exit swirl C_theta3 = U (1 - R - psi/2);
      h03 = h0 - work; p03 from the first-guess stage efficiency;
      at the exit Mach limit C3 = M a3 with T3 = T(h03 - C3^2/2), so C_x = sqrt(C3^2 - C_theta3^2);
      continuity A = m / (rho3 C_x) = 2 pi r_m h  ->  hub/tip = (r_m - h/2)/(r_m + h/2).
    A higher hub/tip means a smaller annulus, a faster axial flow and a higher exit Mach number, so this is the upper
    end of the buildable range. Returns None if the exit swirl alone reaches the Mach limit (no annulus works).
    """
    gas = gas or air()
    d = {**load_design_defaults(), **overrides}
    U = math.sqrt(work_J_kg / psi)
    r_m = U / omega_rad_s
    ct3 = U * (1 - reaction - psi / 2)
    h0 = gas.h(T0_in_K)
    h03 = h0 - work_J_kg
    T03 = gas.T_from_h(h03, T0_in_K)
    T03s = gas.T_from_h(h0 - work_J_kg / d["eta_stage_tt_estimate"], T03)
    p03 = p0_in_Pa * gas.p_ratio_isentropic(T0_in_K, T03s)

    def mach_residual(c3):
        T3 = gas.T_from_h(h03 - 0.5 * c3 * c3, T03)
        return c3 - mach_max * math.sqrt(gas.gamma(T3) * gas.R * T3)

    c3 = brentq(mach_residual, 1e-3, math.sqrt(2 * (h03 - gas.h(0.5 * T03))), xtol=1e-9)
    if abs(ct3) >= c3:
        return None
    cx = math.sqrt(c3 * c3 - ct3 * ct3)
    T3 = gas.T_from_h(h03 - 0.5 * c3 * c3, T03)
    rho3 = p03 * gas.p_ratio_isentropic(T03, T3) / (gas.R * T3)
    h = mdot_kg_s / (rho3 * cx) / (2 * math.pi * r_m)
    return (r_m - h / 2) / (r_m + h / 2)


def stage_from_counts(r_tip_m: float, hub_tip: float, n_ngv: int, n_rotor: int, ngv_gauging_deg: float,
                      rotor_gauging_deg: float, rotor_le_deg: float, tip_clearance_height_ratio: float,
                      **overrides) -> AxialStageGeometry:  # fmt: skip
    """Geometry of an EXISTING single axial stage known by its counts and angles (analysis mode, not design mode).

    Used to rebuild a real turbine (e.g. the KJ-66) whose tip radius and blade counts are published but whose hub
    radius and blade angles are not (those are then calibrated against measured maps; scripts/phase4/
    turbine_calibration_murray.py). Construction (all ASSUMPTIONS where marked; preliminary design only):

    * cylindrical annulus: the same hub and tip radius for NGV and rotor, inlet and exit (ASSUMPTION: no flare data);
    * pitch s = 2 pi r_m / N from the blade count, r_m = r_tip (1 + hub/tip) / 2;
    * throat opening o = s cos(g) from the gauging angle g (cos g = o/s; constant blade height);
    * NGV metal inlet angle = axial (0 deg, as `stator_inlet_angle_deg`); rotor metal inlet angle `rotor_le_deg`;
    * axial chord from Zweifel's tangential loading coefficient Z (Dixon & Hall Ch. 3) with the METAL angles:
      c_x = Z s / (2 cos^2(a_out) |tan a_in - tan a_out|); stagger = (inlet + signed gauging)/2;
    * thickness, leading edge and trailing edge from configs/components/axial_turbine_design.yaml rules
      (as `size_axial_stage`); rotor tip clearance = `tip_clearance_height_ratio` x blade height (no floor).
    Angles: degrees from axial, TurboFlow convention (rotor exit negative; pass `rotor_gauging_deg` unsigned).
    """
    d = {**load_design_defaults(), **overrides}
    r_hub = hub_tip * r_tip_m
    r_m, h = 0.5 * (r_tip_m + r_hub), r_tip_m - r_hub
    fr = d["throat_location_fraction"]
    le = [d["stator_inlet_angle_deg"], rotor_le_deg]
    gs = [ngv_gauging_deg, -rotor_gauging_deg]
    geom: dict = {k: [0.0, 0.0] for k in (
        "pitch", "chord", "stagger_angle", "opening", "leading_edge_angle", "leading_edge_wedge_angle",
        "leading_edge_diameter", "trailing_edge_thickness", "maximum_thickness", "tip_clearance",
        "throat_location_fraction")}  # fmt: skip
    for i, n in enumerate((n_ngv, n_rotor)):
        pitch = 2 * math.pi * r_m / n
        ta, tb = math.tan(le[i] * _D2R), math.tan(gs[i] * _D2R)
        cx = 2 * pitch * math.cos(gs[i] * _D2R) ** 2 * abs(ta - tb) / d["zweifel_coefficient"]
        stagger = 0.5 * (le[i] + gs[i])
        chord = cx / math.cos(stagger * _D2R)
        opening = pitch * math.cos(gs[i] * _D2R)
        geom["pitch"][i], geom["chord"][i], geom["stagger_angle"][i], geom["opening"][i] = (
            pitch,
            chord,
            stagger,
            opening,
        )
        geom["leading_edge_angle"][i] = le[i]
        geom["leading_edge_wedge_angle"][i] = d["leading_edge_wedge_angle_deg"]
        geom["leading_edge_diameter"][i] = d["leading_edge_diameter_chord_ratio"] * chord
        geom["trailing_edge_thickness"][i] = max(d["trailing_edge_thickness_opening_ratio"] * opening,
                                                 d["min_trailing_edge_thickness_m"])  # fmt: skip
        geom["maximum_thickness"][i] = _thickness_rule(abs(le[i] - gs[i]), chord)
        geom["tip_clearance"][i] = 0.0 if i == 0 else tip_clearance_height_ratio * h
        geom["throat_location_fraction"][i] = fr
    return AxialStageGeometry(radius_hub_in=[r_hub, r_hub], radius_hub_out=[r_hub, r_hub],
                              radius_tip_in=[r_tip_m, r_tip_m], radius_tip_out=[r_tip_m, r_tip_m], **geom)  # fmt: skip

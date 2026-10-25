"""Idle-EGT investigation (user request 2026-10-25, item B), in the order the user asked for.

The observation: the Phase 3 baseline engine (~150 N, PR 3, TIT 1150 K) gives an EGT of ~654 C at idle (35 % speed),
hotter than the KJ-66's EGT at FULL power (570-590 C, MFR). That comparison mixes two engines, so step 0 builds
KJ-66 replicas and compares like with like.

0. KJ-66 replicas (KJ-66 compressor family; turbine stage choices = baseline phi 0.7 / psi 1.0 / R 0.5 because the
   KJ-66 turbine geometry is not available):
   - Xiang build: 0.22 kg/s, PR 2.2, 117 krpm; TIT solved for 75 N  -> EGT vs 843 K (MFR);
   - Murray (ArtesJet) build: PR 2.54, 128 krpm; W and TIT solved for 92.3 N and SFC 0.154 kg/(N h) -> EGT vs
     590 C (MFR). Then each replica's operating line down to idle (28-35 % speed; JetCat P80/P220 idle ~28 %).
1. Idle compressor and turbine efficiencies vs design, and the sensitivity of idle EGT to them (diagnostic shift of
   the component efficiency at unchanged pressure ratio: `Engine.diag_eta_delta`).
2. Thermocouple: a bare-wire probe normal to the flow reads T_s + r (T_t - T_s), r = 0.68 (Murray 2009, ASSUM),
   evaluated at the turbine exit (static temperature and Mach number from the meanline turbine).
3. Combustion efficiency at low fuel-air ratio (Murray correlated it with FAR; the correlation is not available):
   idle EGT with eta_b = 0.90 / 0.80 at idle only (sensitivity values, not data).
4. Reynolds number: idle/design Reynolds-number ratio of compressor and turbine (mass-flux based,
   Re ~ (m/A) L / mu, fixed geometry) and the DIRECTION of a Reynolds correction. No cited correlation is applied.

Usage:  uv run python scripts/phase4/idle_egt_investigation.py
Output: validation/phase4/idle_egt_investigation.yaml, idle_egt_operating_lines.csv
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import yaml
from scipy.optimize import brentq

from turboml.physics.compressor import air_viscosity
from turboml.physics.engine import Engine, EngineFailure, design_engine, load_engine_config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"
REF = pd.read_csv(
    ROOT / "validation" / "cases" / "microturbine_reference_assumptions.csv", comment="#"
)
RECOVERY = float(REF.loc[REF.quantity == "thermocouple_recovery_factor", "value"].iloc[0])
SPECS = pd.read_csv(ROOT / "validation" / "cases" / "kj66_engine_specs.csv", comment="#")
TURBINE_CHOICE = {
    "turb_phi": 0.7,
    "turb_psi": 1.0,
    "turb_reaction": 0.5,
}  # baseline choices (assumption)
SPEED_FRACS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28]


def spec(src, q):
    return float(SPECS[(SPECS.source_id == src) & (SPECS.quantity == q)].value.iloc[0])


def tc_reading(p) -> float:
    """Bare-wire thermocouple at the turbine exit: T_s + r (T_t - T_s)."""
    return p.T05_static_K + RECOVERY * (p.T05_K - p.T05_static_K)


def point_row(name, frac, p, design_fn=None) -> dict:
    row = {
        "engine": name,
        "n_frac": frac,
        "converged": p.converged,
        "failure_reason": p.failure_reason,
    }
    if p.converged:
        row.update(n_rpm=p.n_rpm, Fn_N=p.Fn_N, thrust_frac=p.Fn_N / design_fn if design_fn else None, W_kg_s=p.W_air_kg_s,
                   far=p.far, Wf_g_s=p.Wf_kg_s * 1e3, pr_c=p.pr_c, eta_c=p.eta_c, eta_t_ts=p.eta_t_ts, tit_K=p.tit_K,
                   EGT_total_K=p.T05_K, EGT_tc_K=tc_reading(p), turbine_exit_mach=p.turbine_exit_mach,
                   V9_m_s=p.V9_m_s)  # fmt: skip
    return row


def operating_line(name, d, cfg) -> list[dict]:
    eng = Engine(d, cfg)
    rows = []
    for f in SPEED_FRACS:
        p = eng.off_design(f * d.design_point.n_rpm)
        rows.append(point_row(name, f, p, d.design_point.Fn_N))
    return rows


def design_for_thrust(inputs, cfg, fn_target):
    def f(tit):
        try:
            return design_engine({**inputs, "tit_K": tit}, cfg).design_point.Fn_N - fn_target
        except EngineFailure:
            return -fn_target

    tit = brentq(f, 850.0, 1350.0, xtol=0.01)
    return design_engine({**inputs, "tit_K": tit}, cfg)


def design_for_thrust_and_sfc(inputs, cfg, fn_target, tsfc_target):
    """W and TIT such that Fn and TSFC match (Fn ~ proportional to W at fixed PR, N, TIT)."""

    def at_tit(tit):
        w = inputs["W_air_kg_s"]
        for _ in range(6):
            d = design_engine({**inputs, "W_air_kg_s": w, "tit_K": tit}, cfg)
            w *= fn_target / d.design_point.Fn_N
            if abs(d.design_point.Fn_N / fn_target - 1) < 1e-5:
                break
        return d

    tit = brentq(
        lambda t: at_tit(t).design_point.TSFC_kg_N_s - tsfc_target, 900.0, 1300.0, xtol=0.01
    )
    return at_tit(tit)


def summary(d, name, spec_egt_K=None):
    p = d.design_point
    out = {"W_kg_s": round(p.W_air_kg_s, 4), "pr_c": round(p.pr_c, 3), "n_rpm": round(p.n_rpm), "Fn_N": round(p.Fn_N, 2),
           "TSFC_kg_N_h": round(p.TSFC_kg_N_s * 3600, 4), "tit_K": round(p.tit_K, 1), "eta_c": round(p.eta_c, 3),
           "eta_t_ts": round(p.eta_t_ts, 3), "EGT_total_K": round(p.T05_K, 1), "EGT_tc_K": round(tc_reading(p), 1),
           "impeller_r2_mm": round(d.compressor_geometry.radius_out_m * 1e3, 2)}  # fmt: skip
    if spec_egt_K:
        out.update(spec_EGT_K=spec_egt_K, EGT_tc_minus_spec_K=round(tc_reading(p) - spec_egt_K, 1))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_engine_config()
    res: dict = {"thermocouple_recovery_factor": RECOVERY}
    lines = []

    # 0. KJ-66 replicas -------------------------------------------------------------------------------
    xi = {"W_air_kg_s": spec("X17", "air_mass_flow"), "pr_c": spec("X17", "compressor_pressure_ratio"),
          "n_rpm": spec("X17", "max_speed"), **TURBINE_CHOICE}  # fmt: skip
    dx = design_for_thrust(xi, cfg, spec("X17", "thrust"))
    res["kj66_xiang_replica"] = summary(dx, "xiang", spec("X17", "max_exhaust_temperature"))
    lines += operating_line("kj66_xiang", dx, cfg)

    pr_m = (1.01325 + spec("M09", "compressor_delivery_pressure_gauge")) / 1.01325
    mu = {"W_air_kg_s": 0.25, "pr_c": pr_m, "n_rpm": spec("M09", "max_speed"), **TURBINE_CHOICE}
    dm = design_for_thrust_and_sfc(mu, cfg, spec("M09", "thrust"), spec("M09", "sfc") / 3600)
    res["kj66_murray_replica"] = summary(dm, "murray", spec("M09", "egt") + 273.15)
    lines += operating_line("kj66_murray", dm, cfg)

    db = design_engine(cfg["baseline_design"], cfg)
    res["baseline_phase3"] = summary(db, "baseline")
    lines += operating_line("baseline_phase3", db, cfg)
    ol = pd.DataFrame(lines)
    ol.to_csv(OUT / "idle_egt_operating_lines.csv", index=False)
    idle = {}
    for name, g in ol.groupby("engine"):
        g = g[g.converged]
        des = g[g.n_frac == 1.0].iloc[0]
        lo = g[g.n_frac == g.n_frac.min()].iloc[0]
        at35 = g[g.n_frac == 0.35]
        idle[name] = {"lowest_converged_speed_frac": float(lo.n_frac), "EGT_tc_design_K": round(des.EGT_tc_K, 1),
                      "EGT_tc_35pct_K": round(float(at35.EGT_tc_K.iloc[0]), 1) if len(at35) else None,
                      "EGT_tc_lowest_K": round(lo.EGT_tc_K, 1), "thrust_frac_lowest": round(lo.thrust_frac, 4),
                      "idle_over_design_EGT_tc_K_ratio": round(lo.EGT_tc_K / des.EGT_tc_K, 3),
                      "eta_c_design_lowest": [round(des.eta_c, 3), round(lo.eta_c, 3)],
                      "eta_t_design_lowest": [round(des.eta_t_ts, 3), round(lo.eta_t_ts, 3)],
                      "tit_design_lowest_K": [round(des.tit_K, 1), round(lo.tit_K, 1)]}  # fmt: skip
    res["operating_lines_summary"] = idle

    # 1. efficiency sensitivity at idle (baseline engine, 35 % speed) ------------------------------------
    sens = {}
    for comp in ("compressor", "turbine"):
        for de in (-0.02, 0.02):
            e = Engine(db, cfg)
            e.diag_eta_delta = {comp: de}
            p = e.off_design(0.35 * db.design_point.n_rpm)
            sens[f"{comp}_{de:+.2f}"] = round(tc_reading(p), 1) if p.converged else p.failure_reason
    p0 = Engine(db, cfg).off_design(0.35 * db.design_point.n_rpm)
    sens["nominal"] = round(tc_reading(p0), 1)
    res["step1_idle_eta_sensitivity_baseline_35pct_EGT_tc_K"] = sens
    res["step1_dEGT_per_point_eta_K"] = {
        c: round((sens[f"{c}_+0.02"] - sens[f"{c}_-0.02"]) / 4, 2)
        for c in ("compressor", "turbine") if isinstance(sens[f"{c}_+0.02"], float) and isinstance(sens[f"{c}_-0.02"], float)
    }  # fmt: skip

    # 2. thermocouple correction (total - reading) at design and idle ---------------------------------
    res["step2_thermocouple_correction_K"] = {
        name: {"design": round(float(g[g.n_frac == 1.0].EGT_total_K.iloc[0] - g[g.n_frac == 1.0].EGT_tc_K.iloc[0]), 1),
               "35pct": round(float(g[g.n_frac == 0.35].EGT_total_K.iloc[0] - g[g.n_frac == 0.35].EGT_tc_K.iloc[0]), 1)}
        for name, g in ol[ol.converged].groupby("engine")
    }  # fmt: skip

    # 3. combustion efficiency at idle only -----------------------------------------------------------
    comb = {}
    for eb in (0.95, 0.90, 0.80):
        e = Engine(db, cfg)
        e.losses = copy.copy(e.losses)
        p_des = e.off_design(db.design_point.n_rpm)  # continuation start at the design value
        e.losses.eta_burner = eb
        p = e.off_design(0.35 * db.design_point.n_rpm)
        comb[f"eta_b_{eb}"] = {"EGT_tc_K": round(tc_reading(p), 2), "tit_K": round(p.tit_K, 2), "Wf_g_s": round(p.Wf_kg_s * 1e3, 4),
                               "design_check_Fn_N": round(p_des.Fn_N, 3)}  # fmt: skip
    res["step3_combustion_efficiency_at_idle_baseline_35pct"] = comb

    # 4. Reynolds-number ratio idle / design (mass-flux based, fixed geometry) ------------------------
    g = ol[(ol.engine == "baseline_phase3") & ol.converged]
    des, idl = g[g.n_frac == 1.0].iloc[0], g[g.n_frac == 0.35].iloc[0]
    T05d = des.EGT_total_K
    T05i = idl.EGT_total_K
    res["step4_reynolds_ratio_idle_over_design_baseline"] = {
        "compressor (inlet, same T01)": round(idl.W_kg_s / des.W_kg_s, 3),
        "turbine (exit)": round(
            (idl.W_kg_s / des.W_kg_s) * air_viscosity(T05d) / air_viscosity(T05i), 3
        ),
        "direction": "lower Re at idle -> higher loss -> LOWER idle efficiencies -> HIGHER idle EGT (step 1 sensitivity)",
    }
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "idle_egt_investigation.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

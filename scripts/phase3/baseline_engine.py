"""Phase 3: baseline coupled-engine design, off-design points, operating line and plausibility checks.

1. Design the baseline engine (configs/engine_baseline.yaml `baseline_design`).
2. Off-design points (design check, M0.5 at 3 km, part speeds, idle) and the SLS operating line
   (35-100 % speed), all physics-evaluated with the coupled model.
3. Plausibility against the engine databases (review table, JetCat catalog): thrust class, specific
   thrust, TSFC, shaft speed, diameter, length, mass, EGT, idle ratios.
4. Consistency cross-check of the design point against the 0-D cycle (cycle_simple) and pyCycle-CEA,
   both run with the component efficiencies the coupled model produced.
5. Design card (validation/phase3/baseline_design_card.md) with the scope/safety note.

Usage:  uv run python scripts/phase3/baseline_engine.py
Output: validation/phase3/baseline_*.csv/yaml/md
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from turboml.gas import thermo
from turboml.physics import crosscheck
from turboml.physics.cycle_common import CycleInputs
from turboml.physics.engine import Engine, design_card, design_engine, load_engine_config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase3"


def plain(o):
    """numpy scalars / containers -> plain Python (for YAML)."""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def turbine_eta_tt(dp) -> float:
    """Total-to-total turbine efficiency from the coupled design point (for the 0-D models)."""
    gas = thermo.products(dp.far)
    T5s = gas.T_isentropic(dp.tit_K, dp.p05_Pa / dp.p04_Pa)
    return (gas.h(dp.tit_K) - gas.h(dp.T05_K)) / (gas.h(dp.tit_K) - gas.h(T5s))


def plausibility(dp, env, op_line: pd.DataFrame) -> dict:
    rev = pd.read_csv(ROOT / "validation" / "cases" / "review_engine_table.csv")
    jc = pd.read_csv(ROOT / "validation" / "cases" / "jetcat_pro_catalog_2024.csv")
    F = dp.Fn_N
    near = rev[(rev.thrust_N > 0.7 * F) & (rev.thrust_N < 1.3 * F)]
    tsfc_db = (near.fuel_g_min / 60.0 / near.thrust_N * 1e3).dropna()  # g/(kN s)
    rpm_trend = 7.3e5 * F**-0.343
    d_fit = np.polyfit(np.log(rev.dropna(subset=["diameter_mm"]).query("thrust_N >= 50").thrust_N),
                       np.log(rev.dropna(subset=["diameter_mm"]).query("thrust_N >= 50").diameter_mm), 1)  # fmt: skip
    idle = op_line[op_line.converged].sort_values("n_rpm").iloc[0]

    def rng(s):
        s = s.dropna()
        return [
            round(float(s.min()), 3),
            round(float(s.median()), 3),
            round(float(s.max()), 3),
            int(s.size),
        ]

    return {
        "comparison_set": f"review table engines with thrust {0.7 * F:.0f}-{1.3 * F:.0f} N (n = {len(near)})",
        "thrust_N": round(F, 1),
        "specific_thrust_N_s_kg": {"model": round(dp.specific_thrust_N_s_kg, 1),
                                   "database_range_lt_250N": [352, 511], "source": "configs/bounds.yaml database check"},
        "TSFC_g_kN_s": {"model": round(dp.TSFC_kg_N_s * 1e6, 2), "db_min_median_max_n": rng(tsfc_db),
                        "jetcat_pro_2024_min_max": [round(float(jc.sfc_kg_N_h.min() / 3.6 * 1e3), 1),
                                                    round(float(jc.sfc_kg_N_h.max() / 3.6 * 1e3), 1)]},
        "shaft_speed_rpm": {"model": round(dp.n_rpm), "db_trend_7.3e5_F^-0.343": round(rpm_trend),
                            "db_min_median_max_n": rng(near.max_rpm)},
        "diameter_mm": {"model_proxy": round(env["diameter_m"] * 1e3, 1),
                        "db_trend": round(float(np.exp(d_fit[1]) * F ** d_fit[0]), 1), "db_min_median_max_n": rng(near.diameter_mm)},
        "length_mm": {"model_proxy": round(env["length_m"] * 1e3, 1), "db_min_median_max_n": rng(near.length_mm)},
        "dry_mass_kg": {"model_proxy": round(env["dry_mass_kg"], 2), "note": "k D^2 L fitted on JetCat PRO (250-1100 N)"},
        "EGT_C": {"model": round(dp.T05_K - 273.15, 1), "jetcat_max_egt_C": [float(jc.egt_max_C.min()), float(jc.egt_max_C.max())],
                  "db_min_median_max_n": rng(near.max_egt_C)},
        "idle": {"model_lowest_converged_speed_frac": round(idle.n_rpm / dp.n_rpm, 3),
                 "model_thrust_frac": round(idle.Fn_N / dp.Fn_N, 4), "model_EGT_C": round(idle.T05_K - 273.15, 1),
                 "jetcat_idle_rpm_frac": rng(jc.idle_rpm / jc.max_rpm), "jetcat_idle_thrust_frac": rng(jc.idle_thrust_N / jc.thrust_N),
                 "jetcat_egt_min_C": rng(jc.egt_min_C)},
    }  # fmt: skip


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_engine_config()
    t0 = time.perf_counter()
    d = design_engine(cfg["baseline_design"], cfg)
    t_design = time.perf_counter() - t0
    eng = Engine(d, cfg)
    pts = []
    for op in cfg["off_design_points"]:
        p = eng.off_design(op["n_frac"] * d.design_point.n_rpm, op["alt_m"], op["mach"])
        pts.append((op["name"], p))
    rows = [{"point": n, **p.to_dict()} for n, p in pts]
    pd.DataFrame(rows).to_csv(OUT / "baseline_off_design.csv", index=False)

    line = []
    for f in np.round(np.arange(1.0, 0.299, -0.05), 3):
        p = eng.off_design(f * d.design_point.n_rpm)
        line.append({"n_frac": f, **p.to_dict()})
    op_line = pd.DataFrame(line)
    op_line.to_csv(OUT / "baseline_operating_line.csv", index=False)

    # 0-D consistency cross-check at the design point
    dp, L = d.design_point, eng.losses
    ci = CycleInputs(W_air_kg_s=dp.W_air_kg_s, pr_c=dp.pr_c, eta_c=dp.eta_c, tit_K=dp.tit_K, eta_t=turbine_eta_tt(dp),
                     eta_mech=L.eta_mech, eta_burner=L.eta_burner, dp_burner_frac=L.dp_burner_frac,
                     inlet_recovery=L.inlet_recovery, dp_nozzle_frac=L.dp_nozzle_frac, nozzle_cv=L.nozzle_cv,
                     lhv_J_kg=L.lhv_J_kg)  # fmt: skip
    xc = crosscheck.run([ci], n_jobs=1).iloc[0]
    cross = {"equivalent_0D_inputs": {k: round(float(v), 5) for k, v in ci.__dict__.items() if isinstance(v, float)},
             "coupled": {"Fn_N": round(dp.Fn_N, 3), "TSFC_g_kN_s": round(dp.TSFC_kg_N_s * 1e6, 4)},
             "cycle_simple": {"Fn_N": round(float(xc.Fn_N_simple), 3), "TSFC_g_kN_s": round(float(xc.TSFC_kg_N_s_simple) * 1e6, 4)},
             "pycycle_cea": {"Fn_N": round(float(xc.Fn_N_pyc), 3), "TSFC_g_kN_s": round(float(xc.TSFC_kg_N_s_pyc) * 1e6, 4)}}  # fmt: skip
    cross["coupled_vs_simple_pct"] = {
        "Fn": round(100 * (dp.Fn_N / float(xc.Fn_N_simple) - 1), 3),
        "TSFC": round(100 * (dp.TSFC_kg_N_s / float(xc.TSFC_kg_N_s_simple) - 1), 3),
    }
    cross["coupled_vs_pycycle_pct"] = {
        "Fn": round(100 * (dp.Fn_N / float(xc.Fn_N_pyc) - 1), 3),
        "TSFC": round(100 * (dp.TSFC_kg_N_s / float(xc.TSFC_kg_N_s_pyc) - 1), 3),
    }

    env = d.mechanical.checks["envelope"]
    summary = {
        "design_inputs": {k: v for k, v in d.inputs.items()},
        "design_time_s": round(t_design, 1),
        "design_point": {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                         for k, v in dp.to_dict().items() if k not in ("iterations", "max_residual", "time_s")},
        "turbine_sizing": {"r_mean_mm": round(d.turbine_sizing.r_mean_m * 1e3, 2), "U_m_s": round(d.turbine_sizing.U_m_s, 1),
                           "angles_deg": {k: round(v, 2) for k, v in d.turbine_sizing.angles_deg.items()},
                           "warnings": d.notes},
        "mechanical": {"feasible": d.mechanical.feasible, "material": d.mechanical.material,
                       "premium_material_required": d.mechanical.premium_material_required, "tags": d.mechanical.tags,
                       "infeasible_reasons": d.mechanical.infeasible_reasons},
        "off_design": {n: {"converged": bool(p.converged), "failure_reason": p.failure_reason, "Fn_N": round(p.Fn_N, 2),
                           "TSFC_g_kN_s": round(p.TSFC_kg_N_s * 1e6, 2), "W_air_kg_s": round(p.W_air_kg_s, 4),
                           "pr_c": round(p.pr_c, 3), "tit_K": round(p.tit_K, 1), "EGT_K": round(p.T05_K, 1),
                           "surge_flag": p.surge_flag, "time_s": round(p.time_s, 2), "iterations": p.iterations}
                       for n, p in pts},
        "operating_line_converged": f"{int(op_line.converged.sum())}/{len(op_line)}",
        "operating_line_failures": op_line.loc[~op_line.converged, ["n_frac", "failure_reason"]].to_dict("records"),
        "plausibility": plausibility(dp, env, op_line),
        "cross_check_design_point": cross,
    }  # fmt: skip
    summary = plain(summary)
    (OUT / "baseline_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, default_flow_style=None)
    )
    (OUT / "baseline_design_card.md").write_text(
        design_card(d, pts, title="TurboML baseline engine (Phase 3)")
    )
    print(yaml.safe_dump(summary, sort_keys=False, default_flow_style=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

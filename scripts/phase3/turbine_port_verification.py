"""Phase 3: verify the fast axial-turbine meanline model against TurboFlow across the design space.

Acceptance (user decision 2026-10-25): eta_ts within +-2 points, mass flow within +-2 %, specific work
within +-2 %, on >= 200 Sobol samples spread across the design space, plus the Kofskey case
(scripts/phase3/kofskey_fast_turbine.py).

Sampling (scrambled Sobol, seed from configs/default.yaml), 12 dimensions:
  stage design: flow coefficient phi 0.35-0.9, loading psi 0.9-2.2, reaction 0.2-0.6, aspect ratio
  0.8-2.0, rotor tip clearance / height 0.005-0.05, specific work 80-250 kJ/kg;
  design point: mass flow 0.08-1.9 kg/s (log-uniform, configs/bounds.yaml), shaft speed = database trend
  rpm ~ 7.3e5 F^-0.343 (F = 550 N s/kg x mass flow; configs/bounds.yaml) x 0.75-1.3, clipped to the
  50-220 krpm bounds; turbine inlet T0 1000-1350 K (bounds), p0 1.9-4.8 bar;
  off-design: speed fraction 0.6-1.1 and pressure ratio factor 0.7-1.5 on the sizing estimate.
Each sample is evaluated at two points: (a) design speed, estimated design pressure ratio; (b) the
off-design point. Both models run on air (TurboFlow needs a CoolProp fluid). The port comparison uses
incidence_sign="turboflow" (reproduces TurboFlow 0.1.18 exactly); the result with TurboFlow's intended rotor sign
("rotor_plus") is stored alongside to show the effect of that choice.

Usage:  uv run python scripts/phase3/turbine_port_verification.py [--n 256]
Output: validation/phase3/turbine_port_sobol.csv, turbine_port_summary.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import qmc

from turboml.compute import n_workers
from turboml.config import load_config
from turboml.physics.turbine import AxialTurbine
from turboml.physics.turbine_design import size_axial_stage
from turboml.physics.turbine_meanline import AxialTurbineMeanline

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase3"
NAMES = ["phi", "psi", "reaction", "aspect_ratio", "tip_clearance_height_ratio", "work_J_kg",
         "log_mdot", "speed_factor", "T0_in_K", "p0_in_Pa", "speed_frac", "pr_factor"]  # fmt: skip
LO = [0.35, 0.9, 0.2, 0.8, 0.005, 80e3, math.log(0.08), 0.75, 1000.0, 1.9e5, 0.6, 0.7]
HI = [0.9, 2.2, 0.6, 2.0, 0.05, 250e3, math.log(1.9), 1.3, 1350.0, 4.8e5, 1.1, 1.5]
ACCEPT = {"eta_ts_pts": 2.0, "mass_flow_pct": 2.0, "work_pct": 2.0}


def run_sample(i: int, x: dict) -> list[dict]:
    mdot = math.exp(x["log_mdot"])
    rpm = float(np.clip(7.3e5 * (550.0 * mdot) ** -0.343 * x["speed_factor"], 50e3, 220e3))
    omega = rpm * 2 * math.pi / 60
    base = {"sample": i, **x, "mdot_design_kg_s": mdot, "rpm_design": rpm}
    try:
        sz = size_axial_stage(mdot, x["T0_in_K"], x["p0_in_Pa"], omega, x["work_J_kg"], x["phi"], x["psi"],
                              x["reaction"], aspect_ratio=x["aspect_ratio"],
                              tip_clearance_height_ratio=x["tip_clearance_height_ratio"])  # fmt: skip
    except (ValueError, ZeroDivisionError) as e:
        return [{**base, "point": "sizing", "status": f"sizing_failed: {e}"}]
    g = sz.geometry
    geo = {"r_mean_m": sz.r_mean_m, "U_m_s": sz.U_m_s, "hub_tip_min": min(g.radius_hub_out[1] / g.radius_tip_out[1],
           g.radius_hub_in[0] / g.radius_tip_in[0]), "stator_exit_mach_est": sz.mach["stator_exit"],
           "pr_ts_estimate": sz.pr_ts_estimate, "n_warnings": len(sz.warnings), "warnings": "; ".join(sz.warnings)}  # fmt: skip
    tf = AxialTurbine.from_geometry(g.to_turboflow(), x["T0_in_K"], x["p0_in_Pa"], omega)
    fast = AxialTurbineMeanline(g, incidence_sign="turboflow", warm_start=False)
    corr = AxialTurbineMeanline(g, incidence_sign="rotor_plus", warm_start=False)
    rows = []
    for point, sf, pr in (
        ("design", 1.0, sz.pr_ts_estimate),
        ("off_design", x["speed_frac"], sz.pr_ts_estimate * x["pr_factor"]),
    ):
        pr = max(pr, 1.05)
        row = {**base, **geo, "point": point, "speed_frac_eval": sf, "pr_ts_eval": pr}
        t0 = time.perf_counter()
        r_tf = tf.evaluate([tf.operation_point(pr, sf)]).iloc[0]
        row["tf_time_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = fast.evaluate(x["T0_in_K"], x["p0_in_Pa"], omega * sf, pr)
        row["fast_time_ms"] = (time.perf_counter() - t0) * 1e3
        pc = corr.evaluate(x["T0_in_K"], x["p0_in_Pa"], omega * sf, pr)
        row.update({"tf_converged": bool(r_tf.converged), "tf_failure": r_tf.get("failure_reason", ""), "fast_converged": p.converged,
                    "fast_failure": p.failure_reason, "fast_choked": p.choked})  # fmt: skip
        if r_tf.converged:
            row.update({"tf_mdot": r_tf.mass_flow_kg_s, "tf_eta_ts": r_tf.eta_ts,
                        "tf_work": r_tf.power_W / r_tf.mass_flow_kg_s})  # fmt: skip
        if p.converged:
            row.update({"fast_mdot": p.mass_flow_kg_s, "fast_eta_ts": p.eta_ts, "fast_work": p.work_J_kg,
                        "fast_reaction": p.reaction, "fast_stator_exit_mach": p.stator_exit_mach,
                        "fast_rotor_incidence_deg": p.rotor_incidence_deg})  # fmt: skip
        if pc.converged:
            row.update(
                {
                    "corr_mdot": pc.mass_flow_kg_s,
                    "corr_eta_ts": pc.eta_ts,
                    "corr_work": pc.work_J_kg,
                }
            )
        if r_tf.converged and p.converged:
            row["d_mdot_pct"] = 100 * (p.mass_flow_kg_s / r_tf.mass_flow_kg_s - 1)
            row["d_eta_ts_pts"] = 100 * (p.eta_ts - r_tf.eta_ts)
            row["d_work_pct"] = 100 * (p.work_J_kg / row["tf_work"] - 1)
            row["status"] = "compared"
        else:
            row["status"] = (
                "tf_failed"
                if not r_tf.converged and p.converged
                else ("fast_failed" if r_tf.converged else "both_failed")
            )
        if pc.converged and p.converged:
            row["corr_minus_tf_sign_eta_pts"] = 100 * (pc.eta_ts - p.eta_ts)
            row["corr_minus_tf_sign_mdot_pct"] = 100 * (pc.mass_flow_kg_s / p.mass_flow_kg_s - 1)
        rows.append(row)
    return rows


def run_sample_safe(i: int, x: dict) -> list[dict]:
    """run_sample, but an unexpected exception becomes a labelled row instead of killing the run."""
    try:
        return run_sample(i, x)
    except Exception as e:  # noqa: BLE001
        return [
            {
                "sample": i,
                **x,
                "point": "error",
                "status": f"script_error: {type(e).__name__}: {e}"[:200],
            }
        ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    X = qmc.scale(qmc.Sobol(d=len(NAMES), scramble=True, seed=cfg.seed).random(args.n), LO, HI)
    samples = [dict(zip(NAMES, map(float, row), strict=True)) for row in X]
    t0 = time.time()
    # incremental + resumable: each finished sample is appended to a JSON-lines file at once
    part = OUT / "turbine_port_sobol.partial.jsonl"
    done = set()
    if part.exists():
        done = {
            json.loads(ln)["sample"]
            for ln in part.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
    todo = [(i, x) for i, x in enumerate(samples) if i not in done]
    print(f"{len(done)} samples already done, {len(todo)} to run", flush=True)
    with part.open("a", encoding="utf-8") as fh:
        gen = Parallel(n_jobs=n_workers(cfg.compute.n_workers), return_as="generator_unordered")(
            delayed(run_sample_safe)(i, x) for i, x in todo
        )
        for rows in gen:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
    df = pd.DataFrame(
        [json.loads(ln) for ln in part.read_text(encoding="utf-8").splitlines() if ln.strip()]
    )
    df = df.sort_values(["sample", "point"]).reset_index(drop=True)
    df.to_csv(OUT / "turbine_port_sobol.csv", index=False)

    cmp_ = df[df.status == "compared"]
    summ = {"samples": args.n, "seed": cfg.seed, "points": int((~df.point.isin(["sizing", "error"])).sum()),
            "sizing_failed": int((df.point == "sizing").sum()),
            "status_counts": {k: int(v) for k, v in df.status.value_counts().items()},
            "acceptance": ACCEPT, "runtime_s": round(time.time() - t0, 1),
            "fast_time_ms_median_cold": round(float(df.fast_time_ms.median()), 1),
            "tf_time_s_median": round(float(df.tf_time_s.median()), 2), "errors": {}}  # fmt: skip
    for col, key in (
        ("d_eta_ts_pts", "eta_ts_pts"),
        ("d_mdot_pct", "mass_flow_pct"),
        ("d_work_pct", "work_pct"),
    ):
        e = cmp_[col]
        summ["errors"][key] = {"mean": round(float(e.mean()), 4), "mean_abs": round(float(e.abs().mean()), 4),
                               "p95_abs": round(float(e.abs().quantile(0.95)), 4), "max_abs": round(float(e.abs().max()), 4),
                               "n_outside_acceptance": int((e.abs() > ACCEPT[key]).sum())}  # fmt: skip
    worst = cmp_.assign(score=np.maximum.reduce([cmp_.d_eta_ts_pts.abs() / 2, cmp_.d_mdot_pct.abs() / 2,
                                                 cmp_.d_work_pct.abs() / 2])).nlargest(8, "score")  # fmt: skip
    keep = ["sample", "point", "score", "d_eta_ts_pts", "d_mdot_pct", "d_work_pct", "phi", "psi", "reaction",
            "mdot_design_kg_s", "rpm_design", "T0_in_K", "pr_ts_eval", "speed_frac_eval", "hub_tip_min",
            "fast_choked", "fast_rotor_incidence_deg", "n_warnings"]  # fmt: skip
    summ["worst_cases"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                           for r in worst[keep].to_dict("records")]  # fmt: skip
    summ["accepted"] = all(v["n_outside_acceptance"] == 0 for v in summ["errors"].values())
    if "corr_minus_tf_sign_eta_pts" in df:
        e = df.corr_minus_tf_sign_eta_pts.dropna()
        summ["rotor_plus_minus_turboflow_sign"] = {"eta_ts_pts_mean": round(float(e.mean()), 3),
                                             "eta_ts_pts_min": round(float(e.min()), 3),
                                             "eta_ts_pts_max": round(float(e.max()), 3)}  # fmt: skip
    (OUT / "turbine_port_summary.yaml").write_text(
        yaml.safe_dump(summ, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summ, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

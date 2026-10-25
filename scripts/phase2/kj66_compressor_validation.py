"""Phase 2: KJ-66 compressor validation study.

Reference data
  * MEASURED (engine test): Murray (2009), digitized from Xiang et al. (2016) Figs 6-7
    -> validation/cases/kj66_compressor_exp_murray2009.csv   (70-110 krpm, PR and eta, unpaired)
  * CFD: Xiang et al. (2016) speed lines 80/100/117 krpm; Ling et al. (2007) peaks 80-120 krpm.

Steps
  1. BLIND: nominal geometry/assumptions (configs/components/kj66_compressor.yaml), no tuning.
     Same points also through RadComp (independent loss set) and TurboFlow's impeller.
  2. Speed lines 70...117 krpm with surge (peak-PR) and choke flags.
  3. Slip uncertainty: delta_sigma in {-0.10, -0.05, 0, +0.05, +0.10}.
  4. CALIBRATION (documented): delta_sigma and diffuser Cf fitted to the 90 krpm data ONLY;
     validated on 70/80/100/110 krpm.

Usage:  uv run python scripts/phase2/kj66_compressor_validation.py
Output: validation/phase2/kj66_*.csv and kj66_summary.yaml
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.optimize import least_squares

from turboml.physics import compressor_radcomp
from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase2"
EXP = pd.read_csv(ROOT / "validation" / "cases" / "kj66_compressor_exp_murray2009.csv", comment="#")
CALIB_SPEED = 90000
SPEEDS = [70000, 80000, 90000, 100000, 110000, 117000]
DELTA_SIGMAS = [-0.10, -0.05, 0.0, 0.05, 0.10]
FLOWS = np.linspace(0.05, 0.32, 55)


def model_at_exp(geom, **kw) -> pd.DataFrame:
    c = CentrifugalCompressor(geom, **kw)
    rows = []
    for r in EXP.itertuples():
        p = c.evaluate(r.mass_flow_kg_s, r.speed_rpm)
        model = p.pr_tt if r.quantity == "pressure_ratio" else p.eta_tt
        rows.append(
            {**r._asdict(), "model": model, "converged": p.converged, "failure": p.failure_reason}
        )
    d = pd.DataFrame(rows).drop(columns="Index")
    d["err"] = np.where(d.quantity == "pressure_ratio", d.model / d.value - 1, d.model - d.value)
    return d


def stats(d: pd.DataFrame, speeds=None) -> dict:
    if speeds is not None:
        d = d[d.speed_rpm.isin(speeds)]
    out = {}
    for q, lab in [("pressure_ratio", "PR_rel_err"), ("efficiency_ad", "eta_abs_err_pts")]:
        e = d[d.quantity == q].err * 100
        out[lab] = {
            "mean": round(float(e.mean()), 2),
            "max_abs": round(float(e.abs().max()), 2),
            "n": int(e.size),
        }
    return out


def speed_lines(geom, delta_sigma=0.0, diffuser_cf=0.005) -> pd.DataFrame:
    c = CentrifugalCompressor(geom, delta_sigma=delta_sigma, diffuser_cf=diffuser_cf)
    return pd.concat([c.speed_line(N, FLOWS) for N in SPEEDS], ignore_index=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    geom = CentrifugalCompressorGeometry.from_yaml()
    t0 = time.time()
    summary: dict = {"reference": "Murray 2009 measured, digitized from Xiang et al. 2016 Figs 6-7"}

    # 1. blind --------------------------------------------------------------------------
    blind = model_at_exp(geom)
    blind.to_csv(OUT / "kj66_blind_at_measured.csv", index=False)
    summary["blind_turboml"] = stats(blind)
    rc = []
    for r in EXP.itertuples():
        p = compressor_radcomp.evaluate(geom, r.mass_flow_kg_s, r.speed_rpm)
        v = p["pr_tt"] if r.quantity == "pressure_ratio" else p["eta_tt"]
        rc.append({"speed_rpm": r.speed_rpm, "quantity": r.quantity, "value": r.value, "model": v})
    rc = pd.DataFrame(rc)
    rc["err"] = np.where(
        rc.quantity == "pressure_ratio", rc.model / rc.value - 1, rc.model - rc.value
    )
    rc.to_csv(OUT / "kj66_blind_radcomp_at_measured.csv", index=False)
    summary["blind_radcomp"] = stats(rc.dropna())
    summary["blind_radcomp_failed_points"] = int(rc.model.isna().sum())
    print("blind:", summary["blind_turboml"], "| radcomp:", summary["blind_radcomp"])

    # 2+3. speed lines incl. slip uncertainty ------------------------------------------
    sl = pd.concat(
        Parallel(n_jobs=-1)(delayed(speed_lines)(geom, ds) for ds in DELTA_SIGMAS),
        ignore_index=True,
    )
    sl.to_csv(OUT / "kj66_speedlines_slip_uq.csv", index=False)
    nominal = sl[sl.delta_sigma == 0.0]
    peaks = nominal.loc[nominal.converged].loc[
        lambda d: d.groupby("speed_rpm").pr_tt.transform("max") == d.pr_tt
    ][["speed_rpm", "mass_flow_kg_s", "pr_tt", "eta_tt"]]
    summary["blind_max_pr_per_speed"] = {
        int(r.speed_rpm): [round(r.mass_flow_kg_s, 3), round(r.pr_tt, 3)]
        for r in peaks.itertuples()
    }
    fails = (
        nominal[~nominal.converged]
        .groupby("speed_rpm")
        .failure_reason.agg(lambda s: sorted(set(s)))
    )
    summary["failure_reasons_per_speed"] = {int(k): v for k, v in fails.items()}
    # slip band at measured points
    band = []
    for ds in DELTA_SIGMAS:
        d = model_at_exp(geom, delta_sigma=ds)
        d["delta_sigma"] = ds
        band.append(d)
    band = pd.concat(band, ignore_index=True)
    band.to_csv(OUT / "kj66_slip_uq_at_measured.csv", index=False)
    summary["slip_uq_PR_rel_err_mean_by_delta_sigma"] = {
        float(ds): round(float(g[g.quantity == "pressure_ratio"].err.mean() * 100), 2)
        for ds, g in band.groupby("delta_sigma")
    }

    # 4. calibration on 90 krpm only ---------------------------------------------------
    cal = EXP[EXP.speed_rpm == CALIB_SPEED]

    def resid(z):
        c = CentrifugalCompressor(geom, delta_sigma=z[0], diffuser_cf=z[1])
        out = []
        for r in cal.itertuples():
            p = c.evaluate(r.mass_flow_kg_s, r.speed_rpm)
            if not p.converged:
                out.append(1.0)
            elif r.quantity == "pressure_ratio":
                out.append(p.pr_tt / r.value - 1)
            else:
                out.append((p.eta_tt - r.value) / r.value)
        return out

    sol = least_squares(
        resid, x0=[-0.1, 0.02], bounds=([-0.25, 0.002], [0.10, 0.5]), diff_step=1e-3
    )
    ds_cal, cf_cal = (float(v) for v in sol.x)
    calib = model_at_exp(geom, delta_sigma=ds_cal, diffuser_cf=cf_cal)
    calib.to_csv(OUT / "kj66_calibrated_at_measured.csv", index=False)
    val_speeds = [s for s in EXP.speed_rpm.unique() if s != CALIB_SPEED]
    summary["calibration"] = {
        "fitted_on_speed_rpm": CALIB_SPEED,
        "delta_sigma": round(ds_cal, 4),
        "slip_factor": round(CentrifugalCompressor(geom, delta_sigma=ds_cal).slip_factor(), 4),
        "diffuser_cf": round(cf_cal, 4),
        "at_bound": bool(
            np.any(np.isclose(sol.x, [-0.25, 0.002])) or np.any(np.isclose(sol.x, [0.10, 0.5]))
        ),
        "calibration_speed_errors": stats(calib, [CALIB_SPEED]),
        "validation_speed_errors": stats(calib, val_speeds),
    }
    cal_lines = speed_lines(geom, ds_cal, cf_cal)
    cal_lines.to_csv(OUT / "kj66_speedlines_calibrated.csv", index=False)
    pk = cal_lines[cal_lines.converged].loc[
        lambda d: d.groupby("speed_rpm").eta_tt.transform("max") == d.eta_tt
    ]
    summary["calibrated_peak_eta_point_per_speed"] = {
        int(r.speed_rpm): [round(r.mass_flow_kg_s, 3), round(r.pr_tt, 3), round(r.eta_tt, 3)]
        for r in pk.itertuples()
    }
    surge = cal_lines[cal_lines.surge_criterion != ""]
    summary["calibrated_surge_points"] = {
        int(r.speed_rpm): [round(r.mass_flow_kg_s, 3), r.surge_criterion]
        for r in surge.itertuples()
    }
    choke = cal_lines[cal_lines.converged].groupby("speed_rpm").mass_flow_kg_s.max()
    summary["calibrated_max_converged_flow_per_speed"] = {
        int(k): round(float(v), 3) for k, v in choke.items()
    }
    summary["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "kj66_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

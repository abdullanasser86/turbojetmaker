"""KJ-66 compressor "rebuild" from Xiang's geometry, recalibration, and comparison with all reference data
(user request 2026-10-25, item A).

1. Geometry check: the user's extract of Xiang et al. 2017 (r2 33 mm, inlet hub 6.25 mm, inlet shroud 22.5 mm,
   10 blades, 15 diffuser vanes) against configs/components/kj66_compressor.yaml. Which model inputs come from Xiang,
   which from Ling 2007 (CFD paper), which are digitized/assumed, and which calibration factors are still needed.
2. Recalibration: delta_sigma and diffuser Cf re-fitted on the 90 krpm MEASURED points (same split as Phase 2/3)
   with the current code. Must reproduce configs/components/kj66_compressor_calibration.yaml.
3. Predicted map vs reference data, weighted MEAS > CFD > MFR:
   - MEAS: Murray 2009 (digitized) peak PR / efficiency per speed, 70-110 krpm;
   - CFD: Xiang peak PR 1.54/1.76/1.96 and efficiency 0.73/0.63/0.55 at 80/100/117 krpm; Ling 2007 peaks;
   - MFR: design PR 2.2 at 0.22 kg/s, 117 krpm (Xiang); delivery pressure 1.56 bar gauge (PR ~2.54) at 128 krpm
     (Murray Table 1.2, ArtesJet build: mass flow not given -> compare with the peak PR of the 128 krpm line).

Usage:  uv run python scripts/phase4/kj66_rebuild_check.py
Output: validation/phase4/kj66_rebuild_check.yaml, kj66_rebuild_speedlines.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.optimize import least_squares

from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "validation" / "cases"
OUT = ROOT / "validation" / "phase4"
EXP = pd.read_csv(CASES / "kj66_compressor_exp_murray2009.csv", comment="#")
CFD_X = pd.read_csv(CASES / "kj66_compressor_cfd_xiang2016.csv", comment="#")
CFD_L = pd.read_csv(CASES / "kj66_compressor_cfd_ling2007_peaks.csv", comment="#")
CAL = yaml.safe_load(
    (ROOT / "configs" / "components" / "kj66_compressor_calibration.yaml").read_text(
        encoding="utf-8"
    )
)
SPEEDS = [70000, 80000, 90000, 100000, 110000, 117000, 128000]
FLOWS = np.linspace(0.05, 0.34, 59)

# user's extract of Xiang et al. 2017 Table 1 / thesis Table A-i.1 (2026-10-25)
XIANG_USER = {
    "radius_out_m": 0.033,
    "radius_hub_in_m": 0.00625,
    "radius_tip_in_m": 0.0225,
    "number_of_blades": 10,
}


def line(geom, ds, cf, n):
    c = CentrifugalCompressor(geom, delta_sigma=ds, diffuser_cf=cf)
    return c.speed_line(n, FLOWS)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    geom = CentrifugalCompressorGeometry.from_yaml()
    raw = yaml.safe_load(
        (ROOT / "configs" / "components" / "kj66_compressor.yaml").read_text(encoding="utf-8")
    )["impeller"]
    res: dict = {"geometry_check": {}}
    for k, v in XIANG_USER.items():
        res["geometry_check"][k] = {"model": getattr(geom, k), "xiang_user_extract": v,
                                    "match": bool(np.isclose(getattr(geom, k), v))}  # fmt: skip
    res["geometry_check"]["diffuser"] = ("15 wedge vanes (Xiang); vane geometry NOT published -> modelled as a vaneless "
                                         "diffuser with an effective friction factor Cf (calibrated)")  # fmt: skip
    res["model_inputs_by_origin"] = {k: f"{v['kind']}: {v['source']}" for k, v in raw.items()}
    res["calibration_parameters_removed"] = []
    res["calibration_parameters_still_needed"] = {
        "delta_sigma": "absorbs the unpublished blade exit angle / backsweep (slip); Xiang gives no blade angles",
        "diffuser_cf": "absorbs the unmodelled 15-vane wedge diffuser and 90-degree turn; vane geometry not published",
    }

    # 2. recalibration on 90 krpm (MEAS), same procedure as scripts/phase2/kj66_compressor_validation.py
    cal = EXP[EXP.speed_rpm == 90000]

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
    ds, cf = (float(v) for v in sol.x)
    res["recalibration_90krpm"] = {"delta_sigma": round(ds, 4), "diffuser_cf": round(cf, 4),
                                   "stored_delta_sigma": CAL["delta_sigma"]["value"],
                                   "stored_diffuser_cf": CAL["diffuser_cf"]["value"],
                                   "reproduced": bool(abs(ds - CAL["delta_sigma"]["value"]) < 5e-4
                                                      and abs(cf - CAL["diffuser_cf"]["value"]) < 5e-4)}  # fmt: skip

    # 3. predicted map with the stored calibration
    ds0, cf0 = CAL["delta_sigma"]["value"], CAL["diffuser_cf"]["value"]
    sl = pd.concat(
        Parallel(n_jobs=-1)(delayed(line)(geom, ds0, cf0, n) for n in SPEEDS), ignore_index=True
    )
    sl.to_csv(OUT / "kj66_rebuild_speedlines.csv", index=False)
    ok = sl[sl.converged]
    peaks = {}
    for n, g in ok.groupby("speed_rpm"):
        ip, ie = g.pr_tt.idxmax(), g.eta_tt.idxmax()
        peaks[int(n)] = {"peak_pr": round(float(g.pr_tt[ip]), 3), "at_kg_s": round(float(g.mass_flow_kg_s[ip]), 3),
                         "interior_pr_peak": bool(g.mass_flow_kg_s[ip] > g.mass_flow_kg_s.min()),
                         "peak_eta": round(float(g.eta_tt[ie]), 3), "eta_at_kg_s": round(float(g.mass_flow_kg_s[ie]), 3),
                         "pr_at_peak_eta": round(float(g.pr_tt[ie]), 3),
                         "max_converged_flow_kg_s": round(float(g.mass_flow_kg_s.max()), 3)}  # fmt: skip
    res["model_peaks_calibrated"] = peaks
    meas = {}
    for n, g in EXP.groupby("speed_rpm"):
        pr, eta = g[g.quantity == "pressure_ratio"], g[g.quantity == "efficiency_ad"]
        meas[int(n)] = {"max_pr": float(pr.value.max()) if len(pr) else None,
                        "max_eta": float(eta.value.max()) if len(eta) else None}  # fmt: skip
    cmp_rows = []
    for n in SPEEDS:
        m = peaks.get(n, {})
        row = {
            "speed_rpm": n,
            "model_pr_at_peak_eta": m.get("pr_at_peak_eta"),
            "model_peak_eta": m.get("peak_eta"),
            "model_peak_pr": (
                m.get("peak_pr")
                if m.get("interior_pr_peak")
                else "no interior peak above 0.05 kg/s"
            ),
        }
        if n in meas:
            row.update(meas_max_pr=meas[n]["max_pr"], meas_max_eta=meas[n]["max_eta"])
        x = CFD_X[CFD_X.speed_rpm == n]
        if len(x):
            row.update(
                cfd_xiang_peak_pr=float(x.pressure_ratio.max()),
                cfd_xiang_peak_eta=float(x.efficiency_ad.max()),
            )
        lg = CFD_L[CFD_L.speed_rpm == n]
        if len(lg):
            row.update(
                cfd_ling_peak_pr=float(lg.peak_pr.iloc[0]),
                cfd_ling_peak_eta=float(lg.peak_efficiency_tt.iloc[0]),
            )
        cmp_rows.append(row)
    res["comparison_peaks"] = cmp_rows
    # MFR design point: PR 2.2 at 0.22 kg/s and 117 krpm (Xiang); PR ~2.54 at 128 krpm (Murray Table 1.2)
    c = CentrifugalCompressor(geom, delta_sigma=ds0, diffuser_cf=cf0)
    p = c.evaluate(0.22, 117000)
    res["mfr_design_point_117krpm_0.22kg_s"] = {"spec_pr": 2.2, "model_pr": round(p.pr_tt, 3) if p.converged else None,
                                                "model_eta": round(p.eta_tt, 3) if p.converged else None,
                                                "status": p.failure_reason or "ok", "data_type": "MFR"}  # fmt: skip
    res["mfr_murray_128krpm"] = {"spec_pr": round((1.01325 + 1.56) / 1.01325, 3),
                                 "model_pr_at_peak_eta_128krpm": peaks.get(128000, {}).get("pr_at_peak_eta"),
                                 "model_flow_at_peak_eta_kg_s": peaks.get(128000, {}).get("eta_at_kg_s"), "data_type": "MFR",
                                 "note": "mass flow not given in Murray Table 1.2; compared with the PR at the line's peak-efficiency flow (128 krpm is beyond the measured 70-110 krpm: extrapolation)"}  # fmt: skip
    # efficiency trend 80 -> 117 krpm, model vs Xiang CFD
    res["eta_trend_80_to_117krpm"] = {
        "model_peak_eta": [
            peaks[80000]["peak_eta"],
            peaks[100000]["peak_eta"],
            peaks[117000]["peak_eta"],
        ],
        "xiang_cfd_peak_eta": [0.73, 0.63, 0.55],
        "measured_max_eta_80_100_110": [
            meas[80000]["max_eta"],
            meas[100000]["max_eta"],
            meas[110000]["max_eta"],
        ],
        "model_pr_at_peak_eta": [peaks[n]["pr_at_peak_eta"] for n in (80000, 100000, 117000)],
        "xiang_cfd_peak_pr": [1.54, 1.76, 1.96],
    }
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "kj66_rebuild_check.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

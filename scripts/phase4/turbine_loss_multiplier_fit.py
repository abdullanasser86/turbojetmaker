"""Fit ONE turbine loss multiplier k_loss on Murray's measured (derived) turbine efficiency, with speed lines held out
(user decision 2026-10-27, round 6b; D-064).

Model: the meanline turbine with every row's total loss coefficient multiplied by k_loss
(turboml.physics.turbine_meanline.AxialTurbineMeanline(loss_multiplier=...)). The geometry is held FIXED at each of
the two bracketing KJ-66 variants (round 6c):
  T-A = adopted calibration (fit F: flow + exit velocity),  T-B = flow-only fit (fit Q).
Data: Murray Fig 5.18 efficiency (digitized [MEAS]; DERIVED by Murray from the compressor work, +-0.08 ... +-0.34).
Split as in the geometry calibration: CALIBRATION lines 26/38/50/62 krpm corrected, HELD OUT 32/44/56/68 krpm.
Residual: eta_model/eta_meas - 1, every calibration line weighted equally (1/sqrt(n_line)).
Uncertainty: sigma from the LSQ Jacobian (1 parameter) and a profile of RSS vs k_loss; flow-map errors (Fig 5.16)
are reported too, because a higher loss also changes the flow capacity slightly.

Usage:  uv run python scripts/phase4/turbine_loss_multiplier_fit.py
Output: validation/phase4/turbine_loss_multiplier.yaml/.csv, configs/components/turbine_loss_calibration.yaml
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"
_spec = importlib.util.spec_from_file_location(
    "tcal", ROOT / "scripts" / "phase4" / "turbine_calibration_murray.py"
)
TC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TC)
TURB = yaml.safe_load((OUT / "turbine_calibration_murray.yaml").read_text(encoding="utf-8"))
CAL_LINES, HELD_OUT = (26, 38, 50, 62), (32, 44, 56, 68)


def variant_geometry(name: str):
    sec = TURB["adopted" if name == "T-A" else "alternative_flow_only"]
    c = sec["calibrated"]
    p = (c["hub_tip"]["value"], c["ngv_gauging_deg"]["value"], c["rotor_gauging_deg"]["value"])
    le = sec["rotor_le_deg_zero_incidence_at_D10"]
    kj = TC.KJ
    g = TC.stage_from_counts(kj["rotor_tip_radius_m"]["value"], p[0], kj["n_ngv"]["value"], kj["n_rotor"]["value"],
                             p[1], p[2], le, kj["tip_clearance_height_ratio"]["value"])  # fmt: skip
    return p, (g, le)


def eta_residuals(k, geom) -> np.ndarray:
    d = TC.predict(None, k_loss=float(k[0]), geometry=geom)
    d = d[(d.figure == "5.18") & d.speed_krpm.isin(CAL_LINES)]
    out = []
    for _n, g in d.groupby("speed_krpm"):
        out.extend(np.nan_to_num(g.model / g.meas - 1.0, nan=1.0) / math.sqrt(len(g)))
    return np.asarray(out)


def per_line(d: pd.DataFrame) -> dict:
    out = {}
    for n, g in d.groupby("speed_krpm"):
        f, e = g[g.figure == "5.16"], g[g.figure == "5.18"]
        out[int(n)] = {"set": "calibration" if n in CAL_LINES else ("held_out" if n in HELD_OUT else "flow_only_line"),
                       "eta_err_pts": TC.stats(e.err), "Mcorr_err_pct": TC.stats(f.err),
                       "eta_within_published_uncertainty": f"{int(e.within_pub.sum())}/{len(e)}"}  # fmt: skip
    return out


def main() -> int:
    res, rows = {}, []
    for name in ("T-A", "T-B"):
        p, geom = variant_geometry(name)
        d1 = TC.predict(None, k_loss=1.0, geometry=geom)
        fit = least_squares(
            eta_residuals, [2.0], args=(geom,), bounds=([0.5], [8.0]), diff_step=[1e-3]
        )
        k = float(fit.x[0])
        s2 = float(fit.fun @ fit.fun) / max(fit.fun.size - 1, 1)
        sig = math.sqrt(s2 / float((fit.jac.T @ fit.jac)[0, 0]))
        prof = {}
        for kk in np.round(np.linspace(max(0.5, k - 1.0), k + 1.0, 9), 3):
            r = eta_residuals([kk], geom)
            prof[float(kk)] = round(float(r @ r), 5)
        dk = TC.predict(None, k_loss=k, geometry=geom)
        for lab, dd in (("k=1", d1), (f"k={k:.3f}", dk)):
            rows.append(dd.assign(variant=name, case=lab))
        ho = dk[dk.speed_krpm.isin(HELD_OUT)]
        ho1 = d1[d1.speed_krpm.isin(HELD_OUT)]
        res[name] = {"geometry_hub_tip_ngv_rotor": list(p), "k_loss": round(k, 3), "sigma_1": round(sig, 3),
                     "residual_rms_weighted": round(math.sqrt(s2), 4), "rss_profile": prof,
                     "held_out_eta_err_pts": {"k=1": TC.stats(ho1[ho1.figure == "5.18"].err), "fitted": TC.stats(ho[ho.figure == "5.18"].err)},
                     "held_out_Mcorr_err_pct": {"k=1": TC.stats(ho1[ho1.figure == "5.16"].err), "fitted": TC.stats(ho[ho.figure == "5.16"].err)},
                     "eta_err_pts_le_44krpm_all": {"k=1": TC.stats(d1[(d1.figure == "5.18") & (d1.speed_krpm <= 44)].err),
                                                   "fitted": TC.stats(dk[(dk.figure == "5.18") & (dk.speed_krpm <= 44)].err)},
                     "per_speed_line_fitted": per_line(dk), "per_speed_line_k1": per_line(d1)}  # fmt: skip
    ka, kb = res["T-A"]["k_loss"], res["T-B"]["k_loss"]
    sa, sb = res["T-A"]["sigma_1"], res["T-B"]["sigma_1"]
    centre = 0.5 * (ka + kb)
    lo, hi = min(ka - 2 * sa, kb - 2 * sb), max(ka + 2 * sa, kb + 2 * sb)
    res["proposed_for_v6"] = {"k_loss": round(centre, 3), "band": [round(lo, 3), round(hi, 3)],
                              "rule": "centre = mean of the T-A and T-B fits; band = union of their 2-sigma intervals (geometry spread as part of the turbine model uncertainty)"}  # fmt: skip
    pd.concat(rows, ignore_index=True).to_csv(OUT / "turbine_loss_multiplier.csv", index=False)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "turbine_loss_multiplier.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump({k: v for k, v in res.items()}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Joint refit of the KJ-66 turbine geometry AND the loss multiplier on Murray's measured turbine maps
(user decision 2026-10-27, round 7 item 4; D-068). Values are "calibrated, not measured".

Free parameters: hub/tip, NGV gauging angle, rotor gauging angle, k_loss (geometry construction and published
counts as in D-061: 66 mm, 18 NGV, 24 blades, clearance 1 % of height; rotor LE = zero incidence at D.10, recomputed
with the same k_loss).
Data (fitted): Murray Fig 5.16 corrected flow AND Fig 5.18 efficiency (digitized [MEAS]; efficiency DERIVED by Murray),
calibration lines 26/38/50/62 krpm corrected, held out 32/44/56/68 (+ 20 krpm flow only). Residuals relative
(model/meas - 1); every calibration line and both groups weighted equally. NOT fitted: Appendix D exit velocity V5
(independent check of the exit-velocity vs idle-EGT conflict, D-062).
Solver: robust PR matching (`at_pr_tt_robust`); points without a converged bracket count as residual 1.0.
Starts: near T-A (0.743, 64.8, 43.6, 1.9), near T-B (0.52, 70.9, 67.9, 1.8), and a middle start (0.62, 68, 55, 1.8);
the best weighted RSS is reported as the joint fit, all starts are listed (multiple minima are expected, D-062).

Usage:  uv run python scripts/phase4/turbine_joint_refit.py                 # all starts, then merge
        uv run python scripts/phase4/turbine_joint_refit.py --start near_T-B   # one start (parallel runs) ...
        uv run python scripts/phase4/turbine_joint_refit.py --merge            # ... then merge
Output: validation/phase4/turbine_joint_refit.yaml (+ per-start _<start>.yaml/.csv)
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"
_spec = importlib.util.spec_from_file_location(
    "tcal", ROOT / "scripts" / "phase4" / "turbine_calibration_murray.py"
)
TC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TC)
NAMES = ("hub_tip", "ngv_gauging_deg", "rotor_gauging_deg", "k_loss")
LB, UB = (0.40, 50.0, 35.0, 0.8), (0.90, 82.0, 78.0, 3.0)
STARTS = {
    "near_T-A": (0.743, 64.8, 43.6, 1.9),
    "near_T-B": (0.52, 70.9, 67.9, 1.8),
    "middle": (0.62, 68.0, 55.0, 1.8),
}
HELD_OUT = (20, 32, 44, 56, 68)


def residuals(x) -> np.ndarray:
    d = TC.predict(tuple(x[:3]), k_loss=float(x[3]), robust=True)
    d = d[d.set == "calibration"]
    out = []
    for (_n, _f), g in d.groupby(["speed_krpm", "figure"]):
        out.extend(np.nan_to_num(g.model / g.meas - 1.0, nan=1.0) / math.sqrt(len(g)))
    return np.asarray(out)


def summarize(x, fit=None) -> dict:
    d = TC.predict(tuple(x[:3]), k_loss=float(x[3]), robust=True)
    v = TC.predict_v5(tuple(x[:3]), k_loss=float(x[3]), robust=True)
    ho = d[d.set == "held_out"]
    out = {"params": {n: round(float(val), 4) for n, val in zip(NAMES, x, strict=True)},
           "held_out_flow_err_pct": TC.stats(ho[ho.figure == "5.16"].err), "held_out_eta_err_pts": TC.stats(ho[ho.figure == "5.18"].err),
           "calibration_flow_err_pct": TC.stats(d[(d.set == "calibration") & (d.figure == "5.16")].err),
           "calibration_eta_err_pts": TC.stats(d[(d.set == "calibration") & (d.figure == "5.18")].err),
           "failed_points": int(d.model.isna().sum()),
           "V5_check_not_fitted_err_pct": TC.stats(v.err_pct), "V5_within_published_uncertainty": f"{int(v.within_pub.sum())}/{len(v)}",
           "per_speed_line": TC.per_line(d)}  # fmt: skip
    if fit is not None:
        n, k = fit.fun.size, len(x)
        s2 = float(fit.fun @ fit.fun) / (n - k)
        jtj = fit.jac.T @ fit.jac
        cov = s2 * np.linalg.pinv(jtj)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov / np.outer(sig, sig)
        out.update(rss=round(float(fit.fun @ fit.fun), 5), residual_rms_weighted=round(math.sqrt(s2), 4),
                   sigma_1_jacobian={nm: round(float(sv), 4) for nm, sv in zip(NAMES, sig, strict=True)},
                   jtj_condition_number=float(f"{np.linalg.cond(jtj):.3g}"),
                   correlation={f"{NAMES[i]}~{NAMES[j]}": round(float(corr[i, j]), 3) for i in range(k) for j in range(i + 1, k)},
                   optimizer_status=int(fit.status), at_bound=[bool(min(abs(xi - lo), abs(xi - hi)) < 1e-3 * max(1, abs(hi)))
                                                                for xi, lo, hi in zip(x, LB, UB, strict=True)])  # fmt: skip
    return out, d


def run_start(name: str) -> dict:
    fit = least_squares(residuals, STARTS[name], bounds=(LB, UB), x_scale=[0.05, 2.0, 2.0, 0.1], diff_step=[2e-3] * 4,
                        max_nfev=80)  # fmt: skip
    sm, d = summarize(fit.x, fit)
    d.assign(start=name).to_csv(OUT / f"turbine_joint_refit_{name}.csv", index=False)
    out = {"x0": list(STARTS[name]), **sm}
    out = json.loads(json.dumps(out, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / f"turbine_joint_refit_{name}.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False), encoding="utf-8"
    )
    print(name, sm["params"], "rss", sm["rss"], "held-out eta", sm["held_out_eta_err_pts"], "flow", sm["held_out_flow_err_pct"],
          "V5", sm["V5_check_not_fitted_err_pct"], flush=True)  # fmt: skip
    return out


def merge() -> dict:
    res = {"starts": {}}
    for name in STARTS:
        f = OUT / f"turbine_joint_refit_{name}.yaml"
        if f.exists():
            res["starts"][name] = yaml.safe_load(f.read_text(encoding="utf-8"))
    best = min(res["starts"], key=lambda n: res["starts"][n]["rss"])
    res["joint_fit"] = {"start": best, **res["starts"][best]}
    jp = res["joint_fit"]["params"]
    ta, tb = (0.7433, 64.79, 43.62), (0.5191, 70.91, 67.88)
    dist = {lab: round(float(np.linalg.norm([(jp["hub_tip"] - r[0]) / 0.1, (jp["ngv_gauging_deg"] - r[1]) / 5.0,
                                              (jp["rotor_gauging_deg"] - r[2]) / 10.0])), 2) for lab, r in (("T-A", ta), ("T-B", tb))}  # fmt: skip
    res["closer_to"] = {"scaled_distance (hub/tip per 0.1, NGV per 5 deg, rotor per 10 deg)": dist,
                        "variant": min(dist, key=dist.get), "hobbyist_hint_hub_tip_0.56_distance": round(abs(jp["hub_tip"] - 37 / 66), 3)}  # fmt: skip
    (OUT / "turbine_joint_refit.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(
        yaml.safe_dump(
            {"joint_fit": res["joint_fit"]["params"], "closer_to": res["closer_to"]},
            sort_keys=False,
        )
    )
    return res


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--start", choices=list(STARTS), help="run one start (several can run in parallel)"
    )
    ap.add_argument(
        "--merge", action="store_true", help="merge the per-start results into the joint fit"
    )
    a = ap.parse_args()
    if a.start:
        run_start(a.start)
    if a.merge or not a.start:
        if not a.start and not a.merge:
            for name in STARTS:
                run_start(name)
        merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Nickel 337 (Alloy 713C) rupture data: cross-check against Nickel 393 and the creep model, sanity checks, and the
"missing data or physics?" classification of the infeasible TIT > 1150 K designs (user request 2026-10-27, item 3).

Inputs (all transcribed/digitized from Nickel Institute No. 337, see each file's header):
  validation/cases/ni337_in713c_rupture_summary.csv   p. 7 table, 10/100/1000/10,000 h
  validation/cases/ni337_in713c_rupture_tests.csv     Table II individual tests [MEAS]
  validation/cases/ni337_in713c_properties.csv        AMS 5391 minima, melting range
  validation/cases/ni337_in713c_larson_miller_digitized.csv  Figure 7 (digitized)
Checks:
  1. Nickel 337 vs Nickel 393 at every common (temperature, life).
  2. The user's anchors: 1350 F -> 76 ksi (1000 h), 56 ksi (10,000 h), (97) ksi 100 h = extrapolated.
  3. AMS 5391: as-cast 713C must last >= 30 h at 1800 F / 22 ksi; life predicted by (a) the Table II test at that
     stress, (b) log-log interpolation of the p. 7 table, (c) the digitized LM curve, (d) the creep model.
  4. Larson-Miller consistency: every Table II test and table value vs the digitized Figure 7 curve.
  5. The creep model (turboml.physics.creep, Nickel 393 data) vs the Nickel 337 table at 100 h and 1000 h, and the
     LM-fit fallback at 10 h and 10,000 h.
  6. Phase 4 v4 (and v5 if present): infeasible TIT > 1150 K designs, physics vs missing data
     (turboml.analysis.feasibility).
Usage:  uv run python scripts/phase4/ni337_rupture_crosscheck.py
Output: validation/phase4/ni337_crosscheck.yaml, ni337_high_tit_cause_<dataset>.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from turboml.analysis.feasibility import IN713C_SOLIDUS_K, classify
from turboml.physics.creep import KSI_TO_PA, rupture_curve

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "validation" / "cases"
OUT = ROOT / "validation" / "phase4"
SUM = pd.read_csv(CASE / "ni337_in713c_rupture_summary.csv", comment="#")
TST = pd.read_csv(CASE / "ni337_in713c_rupture_tests.csv", comment="#")
PROP = pd.read_csv(CASE / "ni337_in713c_properties.csv", comment="#").set_index("property")
LM = pd.read_csv(CASE / "ni337_in713c_larson_miller_digitized.csv", comment="#")
N393 = pd.read_csv(CASE / "ni393_cast_rupture.csv", comment="#")


def lm_P(T_F, life_h):
    return (T_F + 460.0) * (20.0 + np.log10(life_h)) / 1e3


def lm_stress_ksi(P):
    g = LM[(LM.curve == "vmvc")].sort_values("P_rankine_1e3")
    lg = np.interp(P, g.P_rankine_1e3, np.log10(g.stress_ksi), left=np.nan, right=np.nan)
    return 10**lg


def lm_life_h(T_F, ksi):
    g = LM[(LM.curve == "vmvc")].sort_values("stress_ksi")
    P = float(np.interp(math.log10(ksi), np.log10(g.stress_ksi), g.P_rankine_1e3))
    return 10 ** (P * 1e3 / (T_F + 460.0) - 20.0), P


def main() -> int:
    res = {}
    # 1. vs Nickel 393
    a = N393[N393.alloy == "in713c"].merge(SUM, on=["T_F", "life_h"], suffixes=("_393", "_337"))
    res["vs_ni393"] = {"common_points": len(a),
                       "rows": [{"T_F": int(r.T_F), "life_h": int(r.life_h), "ksi_393": r.strength_ksi_393, "ksi_337": r.strength_ksi_337,
                                 "diff_pct": round(100 * (r.strength_ksi_337 / r.strength_ksi_393 - 1), 2)} for r in a.itertuples()],
                       "new_in_337": [f"{int(r.T_F)} F / {int(r.life_h)} h: {r.strength_ksi} ksi{' (extrapolated)' if r.extrapolated else ''}"
                                      for r in SUM.merge(N393[N393.alloy == 'in713c'][["T_F", "life_h"]], how="left", indicator=True,
                                                         on=["T_F", "life_h"]).query("_merge == 'left_only'").itertuples()]}  # fmt: skip
    # 2. anchors
    s = SUM.set_index(["T_F", "life_h"])
    res["user_anchors_1350F"] = {
        "1000h_ksi": float(s.loc[(1350, 1000), "strength_ksi"]), "10000h_ksi": float(s.loc[(1350, 10000), "strength_ksi"]),
        "100h_ksi": float(s.loc[(1350, 100), "strength_ksi"]), "100h_extrapolated": bool(s.loc[(1350, 100), "extrapolated"]),
        "match": bool(s.loc[(1350, 1000), "strength_ksi"] == 76 and s.loc[(1350, 10000), "strength_ksi"] == 56
                      and s.loc[(1350, 100), "strength_ksi"] == 97 and s.loc[(1350, 100), "extrapolated"])}  # fmt: skip
    # 3. AMS 5391
    T_F, ksi, need = 1800.0, 22.0, float(PROP.loc["ams5391_rupture_life_min", "value"])
    t_test = TST[(TST.T_F == 1800) & (TST.stress_psi == 22000)].life_h.tolist()
    g = SUM[(SUM.T_F == 1800) & ~SUM.extrapolated].sort_values("life_h")
    t_table = 10 ** float(
        np.interp(-math.log10(ksi), -np.log10(g.strength_ksi), np.log10(g.life_h))
    )
    t_lm, P_lm = lm_life_h(T_F, ksi)
    rc = rupture_curve("in713c")
    T_K = (T_F - 32) * 5 / 9 + 273.15
    lo, hi = 1.0, 5.0  # model: bisection on log10(life) of strength(T, t) = 22 ksi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        sv, _how = rc.strength_Pa(T_K, 10**mid)
        lo, hi = (mid, hi) if (sv is not None and sv > ksi * KSI_TO_PA) else (lo, mid)
    res["ams5391_check"] = {"requirement": f">= {need:g} h at 1800 F / 22 ksi (982 C / 152 MPa), as cast",
                            "table_II_test_h": t_test, "p7_table_loglog_h": round(t_table, 1),
                            "larson_miller_fig7_h": round(t_lm, 1), "P_at_22ksi": round(P_lm, 2),
                            "creep_model_h": round(10**lo, 1),
                            "pass": bool(min([*t_test, t_table, t_lm, 10**lo]) >= need)}  # fmt: skip
    # 4. LM consistency
    rows = []
    for r in TST[~TST.runout].itertuples():
        P = lm_P(r.T_F, r.life_h)
        rows.append({"source": "table_II", "T_F": r.T_F, "life_h": r.life_h, "ksi": r.stress_psi / 1e3, "P": P,
                     "curve_ksi": lm_stress_ksi(P), "heat_note": r.heat_note if isinstance(r.heat_note, str) else "",
                     "condition": r.condition})  # fmt: skip
    for r in SUM.itertuples():
        P = lm_P(r.T_F, r.life_h)
        rows.append({"source": "p7_table" + ("_extrapolated" if r.extrapolated else ""), "T_F": r.T_F, "life_h": r.life_h,
                     "ksi": r.strength_ksi, "P": P, "curve_ksi": lm_stress_ksi(P), "heat_note": "", "condition": "as_cast"})  # fmt: skip
    c = pd.DataFrame(rows)
    c["dev_pct"] = 100 * (c.ksi / c.curve_ksi - 1)
    res["larson_miller_consistency"] = {
        k: {"n": int(g.dev_pct.notna().sum()), "mean_dev_pct": round(float(g.dev_pct.mean()), 1),
            "rms_dev_pct": round(float((g.dev_pct**2).mean() ** 0.5), 1), "max_abs_dev_pct": round(float(g.dev_pct.abs().max()), 1),
            "outside_curve_range": int(g.dev_pct.isna().sum())}
        for k, g in c.groupby("source")}  # fmt: skip
    res["larson_miller_curve_range_P"] = {"vmvc_solid": [round(float(LM[(LM.curve == "vmvc") & (LM.segment == "solid")].P_rankine_1e3.min()), 2),
                                                         round(float(LM[(LM.curve == "vmvc") & (LM.segment == "solid")].P_rankine_1e3.max()), 2)],
                                          "vmvc_dashed_extrapolated_to": round(float(LM[LM.curve == "vmvc"].P_rankine_1e3.max()), 2)}  # fmt: skip
    # 5. creep model vs Nickel 337 table
    mrows = []
    for r in SUM[~SUM.extrapolated].itertuples():
        sv, how = rc.strength_Pa(r.T_K, float(r.life_h))
        mrows.append({"T_F": int(r.T_F), "life_h": int(r.life_h), "ni337_ksi": r.strength_ksi,
                      "model_ksi": None if sv is None else round(sv / KSI_TO_PA, 2), "method": how,
                      "diff_pct": None if sv is None else round(100 * (sv / KSI_TO_PA / r.strength_ksi - 1), 1)})  # fmt: skip
    res["creep_model_vs_ni337"] = mrows
    # 6. high-TIT classification
    res["solidus_K"] = IN713C_SOLIDUS_K
    for name in ("phase4_v4", "phase4_v5"):
        ddir = ROOT / "datasets" / name
        if not (ddir / "designs.csv").exists():
            continue
        D = pd.read_csv(ddir / "designs.csv")
        P = pd.read_csv(ddir / "points.csv")
        dp = P[P.point == "design"]
        cl = classify(D, dp)
        hi = cl[cl.in_tit_K > 1150.0]
        cl.to_csv(OUT / f"ni337_high_tit_cause_{name}.csv", index=False)
        n_hi = int(((D.in_tit_K > 1150.0) & (D.status == "ok")).sum())
        res[f"high_tit_infeasibility_{name}"] = {
            "designed_engines_TIT_above_1150K": n_hi, "infeasible": len(hi),
            "cause_counts": hi.cause.value_counts().to_dict(),
            "physics_detail_in713c": hi[hi.cause == "physics"].in713c_physics.value_counts().to_dict(),
            "physics_detail_marm247": hi[hi.cause == "physics"].marm247_physics.value_counts().to_dict(),
            "missing_data_detail": hi[hi.cause == "missing_data"].apply(lambda r: f"{r.data_limited_materials}: in713c[{r.in713c_missing_data}] marm247[{r.marm247_missing_data}]", axis=1).value_counts().to_dict(),
            "ngv_metal_T_K_range_missing_data": [round(float(hi[hi.cause == 'missing_data'].ngv_metal_T_K.min()), 1),
                                                 round(float(hi[hi.cause == 'missing_data'].ngv_metal_T_K.max()), 1)] if (hi.cause == "missing_data").any() else None,
            "rotor_metal_T_K_range_all": [round(float(hi.creep_in713c_rotor_metal_T_K.min()), 1), round(float(hi.creep_in713c_rotor_metal_T_K.max()), 1)],
            "in713c_rotor_data_top_K_100h": round(float(rc.tables[100.0][0][-1]), 2),
        }  # fmt: skip
    c.to_csv(OUT / "ni337_larson_miller_consistency.csv", index=False)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "ni337_crosscheck.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""KJ-66 turbine bracketing variants T-A / T-B (user decision 2026-10-27, round 6c).

  T-A  adopted calibration (fit F: Murray flow map + exit velocity V5); idle 60-90 K hot with k_loss = 1.
  T-B  flow-only fit (fit Q: flow map only); reproduces the idle; contradicts V5.
Both are "calibrated, not measured". Each is run in the full KJ-66 replica (scripts/phase4/kj66_replica_idle_egt.py:
real compressor geometry with the R1 calibration, A9 fitted per nozzle run at the top speed) with
  k_loss = 1 (published loss correlations) and k_loss = the variant's own fitted multiplier
  (validation/phase4/turbine_loss_multiplier.yaml).
Key outputs per variant: idle (< 50 krpm) EGT error and TIT, thrust and SFC at the top speed (run 2, 120 krpm) and at
48 krpm, turbine efficiency, air flow. The spread between the variants is part of the turbine model uncertainty.

Hobbyist hint (user, 2026-10-27; "hobbyist CAD, unverified", NOT measured or published): a 3D-printable KJ66 model
(cults3d "KJ66 Model Jet Turbine") lists an "NGV Inner" part of ~37 mm outer diameter -> NGV hub/tip ~ 37/66 = 0.56.
It is only compared with the two variants' hub/tip ratios, never used as data.

Usage:  uv run python scripts/phase4/kj66_turbine_variants.py
Output: validation/phase4/kj66_turbine_variants.yaml/.csv
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"
_spec = importlib.util.spec_from_file_location(
    "rep", ROOT / "scripts" / "phase4" / "kj66_replica_idle_egt.py"
)
REP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REP)
KL = yaml.safe_load((OUT / "turbine_loss_multiplier.yaml").read_text(encoding="utf-8"))
HOBBY_HUB_TIP = 37.0 / 66.0  # hobbyist CAD, unverified


def at_speed(d: pd.DataFrame, run: str, speed: float) -> dict:
    g = d[(d.run == run) & d.model_converged]
    if g.empty:
        return {}
    r = g.iloc[(g.speed_krpm - speed).abs().argsort().iloc[0]]
    return {"speed_krpm": float(r.speed_krpm), "Fn_N": round(float(r.model_Fn_N), 2),
            "TSFC_g_kN_s": round(float(r.model_TSFC_g_kN_s), 2), "eta_t_ts": round(float(r.model_eta_t_ts), 4),
            "TIT_K": round(float(r.model_tit_K), 1), "EGT_err_K": round(float(r.EGT_err_K), 1),
            "air_flow_err_pct": round(float(r.m_err_pct), 1)}  # fmt: skip


def main() -> int:
    res = {
        "hobbyist_hint": {
            "ngv_hub_tip": round(HOBBY_HUB_TIP, 3),
            "label": "hobbyist CAD, unverified (cults3d KJ66 model, NGV inner part ~37 mm)",
        }
    }
    frames = []
    for var, vkey in (("T-A", "F"), ("T-B", "Q")):
        ht = float(REP.turbine_params(vkey)[0])
        res[var] = {
            "hub_tip": ht,
            "hub_tip_distance_to_hobbyist_hint": round(abs(ht - HOBBY_HUB_TIP), 3),
        }
        for klab, k in (
            ("k_loss=1", 1.0),
            (f"k_loss={KL[var]['k_loss']}", float(KL[var]["k_loss"])),
        ):
            d, s = REP.run_all(label=f"{var} {klab}", variant=vkey, k_loss=k)
            frames.append(d.assign(variant=var, k_loss=k))
            v = s.get("validation", {})
            res[var][klab] = {
                "idle_lt_50krpm": {"EGT_err_K_mean": v.get("EGT_err_K_below_50krpm", {}).get("mean"),
                                   "TIT_K_mean": v.get("TIT_model_K_below_50krpm", {}).get("mean")},
                "all_speeds": {"EGT_err_K_mean": v.get("EGT_err_K", {}).get("mean"), "EGT_err_K_rms": v.get("EGT_err_K", {}).get("rms"),
                               "air_flow_err_pct_mean": v.get("m_err_pct", {}).get("mean"), "converged": v.get("converged")},
                "run2_top_120krpm": at_speed(d, "Nozzle Run 2", 120.0),
                "run2_48krpm": at_speed(d, "Nozzle Run 2", 48.0),
                "A9_m2": {r: x.get("A9_fit_m2") for r, x in s.items() if isinstance(x, dict) and "A9_fit_m2" in x},
            }  # fmt: skip
    # spread between the variants (with each variant's own fitted k_loss)
    a, b = res["T-A"][f"k_loss={KL['T-A']['k_loss']}"], res["T-B"][f"k_loss={KL['T-B']['k_loss']}"]
    spread = {}
    for key in ("Fn_N", "TSFC_g_kN_s", "eta_t_ts", "TIT_K"):
        va, vb = a["run2_top_120krpm"].get(key), b["run2_top_120krpm"].get(key)
        if va is not None and vb is not None:
            spread[f"top_{key}"] = {
                "T-A": va,
                "T-B": vb,
                "rel_spread_pct": round(100 * abs(va - vb) / (0.5 * (va + vb)), 2),
            }
    spread["idle_EGT_err_K"] = {
        "T-A": a["idle_lt_50krpm"]["EGT_err_K_mean"],
        "T-B": b["idle_lt_50krpm"]["EGT_err_K_mean"],
    }
    res["variant_spread_with_fitted_k"] = spread
    pd.concat(frames, ignore_index=True).to_csv(OUT / "kj66_turbine_variants.csv", index=False)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "kj66_turbine_variants.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

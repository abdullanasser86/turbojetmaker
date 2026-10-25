"""Pilot feasibility analysis and stop rule (user decision 2026-10-25, item 1).

Reports, for the pilot of a dataset version:
  * the feasible fraction (of all pilot designs, and of the designs that could be built);
  * failure reasons per design and per operating point;
  * the active constraint(s) of every infeasible design;
  * the MAR-M 247 no-data branch (default alloy fails, premium alloy has no rupture data at its metal temperature);
  * whether the CONSERVATIVE SETTINGS are the cause: feasibility recomputed from the stored stresses and temperatures
    with each setting relaxed in turn (life, safety factor, blade taper At/Ah, pattern factor) and all together.
    The relaxed values are sensitivity probes, not recommendations.
Stop rule: if feasible / pilot designs < min_feasible_fraction (configs/dataset_phase4_v2.yaml), stop and report.

Usage:  uv run python scripts/phase4/pilot_feasibility_analysis.py [--config configs/dataset_phase4_v2.yaml]
Output: validation/phase4/pilot_analysis_<name>.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from turboml.config import load_config
from turboml.physics import creep
from turboml.physics.mechanical import MaterialSet

ROOT = Path(__file__).resolve().parents[2]

# sensitivity probes: one setting relaxed at a time, then all together
PROBES = {
    "default (100 h, SF 1.5, At/Ah 1, PF 0.25)": {},
    "life 50 h": {"life_h": 50.0},
    "SF 1.25": {"safety_factor": 1.25},
    "SF 1.0": {"safety_factor": 1.0},
    "At/Ah 0.34 (K 0.67)": {"tip_to_hub_area_ratio": 0.34},
    "PF 0.15": {"pattern_factor": 0.15},
    "all relaxed (50 h, SF 1.0, At/Ah 0.34, PF 0.15)": {"life_h": 50.0, "safety_factor": 1.0,
                                                         "tip_to_hub_area_ratio": 0.34, "pattern_factor": 0.15},
}  # fmt: skip


def recompute(row, P, mats: MaterialSet, tip_limit: float) -> tuple[bool, str]:
    """Feasibility of one designed engine with other creep settings, from the stored raw values."""
    T3 = P.loc[(row.design_id, "design"), "T03_K"]
    active = []
    ok_material = False
    last = None
    for mat in (mats.default, mats.premium):
        k = mats.blade_taper_factor
        r = creep.assess(mat, mats.life_h, mats.safety_factor, row[f"creep_{mat}_rotor_metal_T_K"],
                         k * row[f"creep_{mat}_rotor_stress_centrifugal_Pa"], row[f"creep_{mat}_rotor_stress_gas_bending_Pa"],
                         row[f"creep_{mat}_ngv_stress_gas_bending_Pa"], row.in_tit_K, T3, mats.pattern_factor)  # fmt: skip
        last = r
        if r.feasible:
            ok_material = True
            break
    if not ok_material:
        active += last.active
    if row.impeller_tip_speed_m_s > tip_limit:
        active.append("tip_speed")
    return not active, ";".join(active)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset_phase4_v2.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    d_dir = ROOT / cfg["output_dir"]
    D = pd.read_csv(d_dir / "designs.csv")
    P = pd.read_csv(d_dir / "points.csv").set_index(["design_id", "point"])
    n_pilot = int(cfg.get("pilot_designs", len(D)))
    D = D[D.design_id < n_pilot]
    ok = D[D.status == "ok"].copy()
    feas = ok.feasible.astype(bool)
    res: dict = {"name": cfg["name"], "pilot_designs": len(D), "designed_engines": len(ok),
                 "feasible": int(feas.sum()), "feasible_fraction_of_pilot": round(feas.sum() / len(D), 4),
                 "feasible_fraction_of_designed": round(feas.sum() / max(len(ok), 1), 4),
                 "min_feasible_fraction": cfg.get("min_feasible_fraction")}  # fmt: skip
    res["stop"] = bool(res["feasible_fraction_of_pilot"] < (cfg.get("min_feasible_fraction") or 0))
    res["design_failure_reasons"] = {str(k): int(v) for k, v in D.loc[D.status != "ok", "failure_reason"]
                                     .str.replace(r":.*", "", regex=True).value_counts().items()}  # fmt: skip
    pts = P.reset_index()
    pts = pts[pts.design_id < n_pilot]
    res["point_failure_reasons"] = {
        pt: {str(k): int(v) for k, v in g.loc[~g.converged.astype(bool), "failure_reason"].fillna("").value_counts().items()}
        for pt, g in pts.groupby("point")
    }  # fmt: skip
    res["point_converged_of_designed"] = {
        pt: f"{int(g[g.design_id.isin(ok.design_id)].converged.sum())}/{len(ok)}"
        for pt, g in pts.groupby("point")
    }
    res["infeasible_by_active_constraints"] = {str(k): int(v) for k, v in ok.loc[~feas, "active_constraints"]
                                               .fillna("").value_counts().items()}  # fmt: skip
    res["active_constraint_per_infeasible_design"] = {
        int(r.design_id): r.active_constraints for r in ok[~feas].itertuples()
    }
    single = {}
    for s in ok.loc[~feas, "active_constraints"].fillna(""):
        for c in s.split(";"):
            if c:
                single[c] = single.get(c, 0) + 1
    res["designs_hitting_each_constraint"] = dict(sorted(single.items(), key=lambda kv: -kv[1]))
    if "material_status" in ok:
        res["material_status"] = {
            str(k): int(v) for k, v in ok.material_status.value_counts().items()
        }
        gap = ok[ok.material_status == "unchecked_no_material_data"]
        res["marm247_no_data_branch"] = {
            "count": len(gap),
            "tit_range_K": [round(float(gap.in_tit_K.min()), 1), round(float(gap.in_tit_K.max()), 1)] if len(gap) else None,
        }  # fmt: skip
    sens_cols = [c for c in ok.columns if c.startswith("sens_") and c.endswith("_feasible")]
    res["stored_sensitivity_feasible_counts"] = {
        c[5:-9]: int(ok[c].astype(bool).sum()) for c in sens_cols
    }
    tip_limit = load_config().constraints.max_impeller_tip_speed_m_s
    probes = {}
    for name, ov in PROBES.items():
        mats = MaterialSet.from_yaml(**ov)
        out = [recompute(r, P, mats, tip_limit) for _, r in ok.iterrows()]
        n_feas = sum(f for f, _ in out)
        counts: dict = {}
        for f, a in out:
            if not f:
                for c in a.split(";"):
                    counts[c] = counts.get(c, 0) + 1
        probes[name] = {"feasible": n_feas, "fraction_of_pilot": round(n_feas / len(D), 4),
                        "constraint_hits": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}  # fmt: skip
    res["conservative_settings_probe"] = probes
    res["check_default_probe_reproduces_dataset"] = probes[next(iter(PROBES))]["feasible"] == int(
        feas.sum()
    )
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    out = ROOT / "validation" / "phase4" / f"pilot_analysis_{cfg['name']}.yaml"
    out.write_text(yaml.safe_dump(res, sort_keys=False))
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

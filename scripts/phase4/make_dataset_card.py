"""Write the dataset card (docs/dataset_cards/<name>.md) and failure statistics (validation/phase4/) for a
generated dataset. Run after generate_dataset.py (and crosscheck_dataset.py).

Usage:  uv run python scripts/phase4/make_dataset_card.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from turboml.config import load_config
from turboml.physics.engine import SAFETY_NOTE

ROOT = Path(__file__).resolve().parents[2]


def counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.value_counts().items()}


def md_table(d: dict, h1: str, h2: str = "count") -> list[str]:
    return [f"| {h1} | {h2} |", "|---|---|"] + [f"| {k or '(none)'} | {v} |" for k, v in d.items()]


def main() -> int:
    dcfg = yaml.safe_load((ROOT / "configs" / "dataset_phase4.yaml").read_text())
    name = dcfg["name"]
    out_dir = ROOT / dcfg["output_dir"]
    D = pd.read_csv(out_dir / "designs.csv")
    P = pd.read_csv(out_dir / "points.csv")
    split = pd.read_csv(ROOT / "data" / "splits" / f"{name}_split.csv")
    xc_path = ROOT / "validation" / "phase4" / f"crosscheck_{name}.yaml"
    xc = yaml.safe_load(xc_path.read_text()) if xc_path.exists() else None
    bounds = load_config().bounds
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    n_total, n_done = int(dcfg["n_designs"]), len(D)
    ok = D[D.status == "ok"]

    stats = {"name": name, "designs_evaluated": n_done, "designs_planned": n_total,
             "design_status": counts(D.status),
             "design_failure_reasons": counts(D.loc[D.status != "ok", "failure_reason"].str.replace(r":.*", "", regex=True)),
             "feasible_designs": int(ok.feasible.fillna(False).astype(bool).sum()),
             "infeasible_designs_by_active_constraint": counts(ok.loc[~ok.feasible.astype(bool), "active_constraints"].fillna("")),
             "premium_material_required": int(ok.premium_material_required.fillna(False).astype(bool).sum()),
             "points": {}, "split": counts(split[split.design_id < n_done].split) if n_done < n_total else counts(split.split),
             "split_ok_designs": counts(ok.split), "calibration_sampled": int(D.calibration_sampled.sum()),
             "time_per_design_s": {"mean": round(float(D.time_s.mean()), 2), "median": round(float(D.time_s.median()), 2),
                                   "p95": round(float(D.time_s.quantile(0.95)), 2), "max": round(float(D.time_s.max()), 2)}}  # fmt: skip
    for pt, g in P.groupby("point"):
        g_ok = g[g.design_id.isin(ok.design_id)]
        stats["points"][pt] = {"converged_all_designs": f"{int(g.converged.sum())}/{len(g)}",
                               "converged_designed_engines": f"{int(g_ok.converged.sum())}/{len(g_ok)}",
                               "failure_reasons_designed_engines": counts(g_ok.loc[~g_ok.converged.astype(bool), "failure_reason"].fillna("")),
                               "surge_flags": counts(g_ok.surge_flag.fillna("")) if "surge_flag" in g_ok else {}}  # fmt: skip
    warn = ok.geom_turb_sizing_warnings.fillna("")
    stats["turbine_outside_turboflow_range"] = {
        "rotor_hub_tip": int(warn.str.contains("rotor hub_tip_ratio").sum()),
        "any_warning": int((warn != "").sum()),
        "feasible_with_rotor_hub_tip_warning": int(
            ok.feasible.astype(bool)[warn.str.contains("rotor hub_tip_ratio")].sum()
        ),
    }
    pfm = ok.get("creep_in713c_ngv_pf_max")
    if pfm is not None:
        stats["ngv_pf_max_in713c"] = {
            q: round(float(np.nanquantile(pfm.astype(float), qq)), 3)
            for q, qq in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))
        }
    timing = sorted((ROOT / "runs").glob(f"*_dataset_{name}/timing.json"))
    if timing:
        t = json.loads(timing[-1].read_text())
        stats["last_run"] = t
    sdir = ROOT / "validation" / "phase4"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"dataset_stats_{name}.yaml").write_text(yaml.safe_dump(stats, sort_keys=False))

    pilot = n_done < n_total
    L = [f"# Dataset card: `{name}`{' (PILOT)' if pilot else ''}", "", f"> **{SAFETY_NOTE}**", "",
         f"Status: {n_done} of {n_total} planned designs evaluated" + (" (pilot: the first designs of the Sobol sequence; the full run "
         "is started on the user's laptop after the wall-time estimate)." if pilot else "."), "",
         f"Generated with git {git} by `scripts/phase4/generate_dataset.py` (`turboml/data/generator.py`), config "
         "`configs/dataset_phase4.yaml`. Raw records: `datasets/<name>/designs.jsonl` (git-ignored, regenerate from seed + config).", "",
         "## What one record is", "",
         "One **design** = cycle design variables + turbine stage choices, sized by the coupled physics model (design mode), "
         "checked mechanically (creep, AN², tip speed), then evaluated at 4 **points**: design (SLS), M0.5 at 3 km (design "
         "speed), 80 % speed (SLS), idle 35 % speed (SLS). Every design and every point carries `converged` and "
         "`failure_reason`; failed designs and points are kept, never dropped.", "",
         "Physics: 0-D cycle (`cycle_simple` components) with meanline compressor (KJ-66-calibrated, provisional) and "
         "meanline axial turbine (TurboFlow formulation); NOT surrogate output. Turbine efficiency at micro scale is "
         "unvalidated (flag on every design).", "",
         "## Design variables (Sobol, scrambled, seed " + str(dcfg["seed"]) + ")", "", "| variable | range | scale | source |", "|---|---|---|---|"]  # fmt: skip
    for v, spec in dcfg["design_variables"].items():
        b = bounds[v]
        L.append(
            f"| {spec['key']} | {b.lower:g} – {b.upper:g} | {spec['scale']} | {b.source[:120]} |"
        )
    L += ["", f"Compressor calibration factors: {stats['calibration_sampled']} of {n_done} designs sample (Δσ, Cf) uniformly in the "
          "inflated band (configs/components/kj66_compressor_calibration.yaml); the others use the nominal values.", "",
          "Fixed assumptions: component losses `configs/engine_baseline.yaml` (DePaola values as summarised by the user, "
          "off-design loss scaling with corrected flow²); materials and creep settings `configs/materials.yaml` "
          "(life 100 h, safety factor 1.5, taper factor 1.0, pattern factor NOT sourced → NGV PF_max stored).", "",
          "## Held-out test set (frozen)", "",
          f"`data/splits/{name}_split.csv`: {int(round(dcfg['test_fraction'] * 100))} % of design ids, seeded permutation, written before "
          "any design was evaluated. Never used for training or tuning.", ""]  # fmt: skip
    L += md_table(stats["split"], "split (evaluated designs)") + [""]
    L += ["## Design-level outcome", ""] + md_table(stats["design_status"], "status") + [""]
    L += (
        ["Design-mode failure reasons:", ""]
        + md_table(stats["design_failure_reasons"], "reason")
        + [""]
    )
    L += [f"Mechanically feasible (selected material passes creep; tip speed OK): **{stats['feasible_designs']}** of "
          f"{len(ok)} designed engines; premium material required: {stats['premium_material_required']}.", "",
          "Infeasible designed engines by active constraint(s):", ""] + md_table(stats["infeasible_designs_by_active_constraint"], "active constraints") + [""]  # fmt: skip
    L += [
        "## Point-level outcome (designed engines)",
        "",
        "| point | converged | failure reasons | surge flags |",
        "|---|---|---|---|",
    ]
    for pt, s in stats["points"].items():
        L.append(f"| {pt} | {s['converged_designed_engines']} | {', '.join(f'{k}: {v}' for k, v in s['failure_reasons_designed_engines'].items()) or '–'} | "
                 f"{', '.join(f'{k}: {v}' for k, v in s['surge_flags'].items() if k) or '–'} |")  # fmt: skip
    L += ["", "## Cross-check (pyCycle-CEA, random 2 % of designed engines)", ""]
    if xc:
        L += [f"- checked {xc['checked']} of {xc['designed_engines']}; coupled model vs pyCycle-CEA: flagged "
              f"{xc['coupled_vs_pycycle']['flagged']} (gate passed: {xc['coupled_vs_pycycle']['gate_passed']}), max |ΔFn| "
              f"{xc['coupled_vs_pycycle']['max_abs_Fn_pct']} %, max |ΔTSFC| {xc['coupled_vs_pycycle']['max_abs_TSFC_pct']} %",
              f"- 0-D cycle_simple vs pyCycle-CEA: flagged {xc['simple_vs_pycycle']['flagged']} (gate passed: {xc['simple_vs_pycycle']['gate_passed']})",
              f"- secondary metric |ΔFn|/Fg (labelled secondary): max {xc['secondary_dFn_over_Fg_max_abs']}", ""]  # fmt: skip
    else:
        L += ["Not run yet.", ""]
    t = stats["time_per_design_s"]
    L += [
        "## Cost",
        "",
        f"Per design (one worker): mean {t['mean']} s, median {t['median']} s, 95th percentile {t['p95']} s, max {t['max']} s.",
    ]
    if "last_run" in stats:
        r = stats["last_run"]
        L.append(
            f"Last run: {r['designs_this_run']} designs in {r['wall_s']:.0f} s wall on {r['workers']} workers."
        )
    L += ["", "## Known limitations", "",
          "- Compressor family is KJ-66-like (one geometry family, one calibration); vaneless diffuser model.",
          "- Turbine efficiency at micro scale unvalidated; incidence-sign convention open (D-029).",
          "- Creep check uses typical (not minimum) rupture data; blade taper factor 1 (conservative); NGV pattern factor not sourced.",
          f"- Turbine geometry outside TurboFlow's recommended ranges (kept, warning in `geom_turb_sizing_warnings`): rotor "
          f"hub/tip in {stats['turbine_outside_turboflow_range']['rotor_hub_tip']} of {len(ok)} designed engines "
          f"(feasible among them: {stats['turbine_outside_turboflow_range']['feasible_with_rotor_hub_tip_warning']}); "
          "the loss correlations are less trustworthy there.",
          "- A compressor operating on its choke line can't be resolved (labelled `choke_compressor_*`).",
          "- Design variables are sampled independently in the box: many corners are physically impossible for this "
          "engine family (labelled design failures).", ""]  # fmt: skip
    cdir = ROOT / "docs" / "dataset_cards"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{name}.md").write_text("\n".join(L))
    print(yaml.safe_dump(stats, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

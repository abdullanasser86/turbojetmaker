"""Write the dataset card (docs/dataset_cards/<name>.md) and failure statistics (validation/phase4/) for a
generated dataset. Run after generate_dataset.py (and crosscheck_dataset.py).

Usage:  uv run python scripts/phase4/make_dataset_card.py
"""

from __future__ import annotations

import argparse
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


def regime_split(D: pd.DataFrame, ok: pd.DataFrame, thr: float | None) -> dict:
    """Feasibility by tip-speed regime (validated <= 415 m/s, extrapolated above) and, if set, by TIT band."""
    out = {}
    bands = [("all TIT", lambda x: x.in_tit_K > 0)]
    if thr is not None:
        bands += [
            (f"TIT <= {thr:g} K", lambda x: x.in_tit_K <= thr),
            (f"TIT > {thr:g} K", lambda x: x.in_tit_K > thr),
        ]
    for reg in ("validated", "extrapolated"):
        for lab, f in bands:
            dm = (D.tip_speed_regime == reg) & f(D)
            om = (ok.tip_speed_regime == reg) & f(ok)
            nf = int(ok[om].feasible.astype(bool).sum())
            out[f"{reg} / {lab}"] = {"designs": int(dm.sum()), "designed": int(om.sum()), "feasible": nf,
                                     "feasible_%_of_designs": round(100 * nf / max(int(dm.sum()), 1), 1),
                                     "feasible_%_of_designed": round(100 * nf / max(int(om.sum()), 1), 1)}  # fmt: skip
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset_phase4_v2.yaml"))
    args = ap.parse_args()
    dcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name = dcfg["name"]
    out_dir = ROOT / dcfg["output_dir"]
    D = pd.read_csv(out_dir / "designs.csv")
    P = pd.read_csv(out_dir / "points.csv")
    split = pd.read_csv(ROOT / "data" / "splits" / f"{name}_split.csv")
    xc_path = ROOT / "validation" / "phase4" / f"crosscheck_{name}.yaml"
    xc = yaml.safe_load(xc_path.read_text(encoding="utf-8")) if xc_path.exists() else None
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
    thr = dcfg.get("high_tit_threshold_K")
    if thr is not None:  # user decision 2026-10-25: feasibility inside / outside the high-TIT band
        split_tit = {}
        for lab, mall in (
            (f"TIT > {thr:g} K", D.in_tit_K > thr),
            (f"TIT <= {thr:g} K", D.in_tit_K <= thr),
        ):
            sel = ok[mall[D.status == "ok"].values] if len(ok) else ok
            nf = int(sel.feasible.astype(bool).sum())
            split_tit[lab] = {"designs": int(mall.sum()), "designed": len(sel), "feasible": nf,
                              "feasible_%_of_designs": round(100 * nf / max(int(mall.sum()), 1), 1),
                              "feasible_%_of_designed": round(100 * nf / max(len(sel), 1), 1)}  # fmt: skip
        stats["high_tit_split"] = split_tit
        hi_inf = ok[(ok.in_tit_K > thr) & ~ok.feasible.astype(bool)]
        stats["infeasible_high_tit_by_constraint"] = counts(
            hi_inf.active_constraints.fillna("") + " (" + hi_inf.material_status.fillna("") + ")"
        )
    if (
        "tip_speed_regime" in D and D.tip_speed_regime.notna().any()
    ):  # R3 (user decision 2026-10-27)
        stats["tip_speed_regime"] = regime_split(D, ok, thr)
    if thr is not None:  # round 5 item 3b: physics or missing data?
        from turboml.analysis.feasibility import classify

        cl = classify(D, P[P.point == "design"])
        hi_c = cl[cl.in_tit_K > thr]
        stats["high_tit_infeasibility_cause"] = {"infeasible": len(hi_c), "cause": counts(hi_c.cause),
                                                 "physics_in713c": counts(hi_c[hi_c.cause == "physics"].in713c_physics),
                                                 "physics_marm247": counts(hi_c[hi_c.cause == "physics"].marm247_physics),
                                                 "missing_data_limited_materials": counts(hi_c[hi_c.cause == "missing_data"].data_limited_materials)}  # fmt: skip
    dpts = P[(P.point == "design") & P.design_id.isin(ok.design_id)]
    if (
        "eta_t_tt" in dpts and dpts.eta_t_tt.notna().any()
    ):  # v6+: turbine efficiency sanity check (D-069)
        e = dpts.set_index("design_id").eta_t_tt
        rng = dcfg.get("turbine_eta_tt_plausible", {}).get("range")
        q = {
            f"p{int(100 * qq)}": round(float(e.quantile(qq)), 4)
            for qq in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        }
        stats["turbine_eta_tt_design"] = {"quantiles": q, "plausible_range": rng}
        if rng:
            bad = e[(e < rng[0]) | (e > rng[1])]
            stats["turbine_eta_tt_design"].update(flagged=len(bad), flagged_feasible=int(ok.set_index("design_id").loc[bad.index].feasible.astype(bool).sum()),
                                                  flagged_ids=[int(i) for i in bad.index[:50]])  # fmt: skip
            pd.DataFrame({"design_id": bad.index, "eta_t_tt": bad.values}).to_csv(
                ROOT / "validation" / "phase4" / f"turbine_eta_flags_{name}.csv", index=False
            )
    if "retries" in P:  # v6+: turbine-solver restarts (D-067)
        pr = {}
        for pt, g in P.groupby("point"):
            pr[pt] = {"converged_on_retry": int(g.converged_on_retry.eq(True).sum()),
                      "failed_after_retries": int((~g.converged.astype(bool) & (g.retries.fillna(0) > 0)).sum()),
                      "turbine_no_convergence": int(g.failure_reason.fillna("").str.startswith("turbine_no_convergence").sum())}  # fmt: skip
        stats["turbine_solver_retries"] = pr
    if (
        "turbine_loss_multiplier" in D and D.turbine_loss_multiplier.notna().any()
    ):  # v6+: k_loss subset (D-064)
        nominal = float(D.turbine_loss_multiplier.mode().iloc[0])
        sub = {}
        for lab, m in (
            ("nominal", D.turbine_loss_multiplier == nominal),
            ("sampled", D.turbine_loss_multiplier != nominal),
        ):
            g = D[m]
            o = g[g.status == "ok"]
            sub[lab] = {"designs": len(g), "designed": len(o), "feasible": int(o.feasible.astype(bool).sum()),
                        "k_loss_range": [round(float(g.turbine_loss_multiplier.min()), 3), round(float(g.turbine_loss_multiplier.max()), 3)]}  # fmt: skip
        stats["k_loss_subset"] = sub
    if "derived_turb_hub_tip_mach_binding" in ok:
        stats["t1_hub_tip"] = {
            "sampled_hub_tip_range": [round(float(ok.derived_turb_hub_tip.min()), 3), round(float(ok.derived_turb_hub_tip.max()), 3)],
            "mach_limit_binding": f"{int(ok.derived_turb_hub_tip_mach_binding.astype(float).sum())}/{len(ok)}",
            "no_buildable_hub_tip_designs": int(D.failure_reason.fillna("").str.startswith("design_turbine_no_buildable_hub_tip").sum()),
        }  # fmt: skip
    if "material_status" in ok:  # dataset versions >= v2
        stats["material_status"] = counts(ok.material_status.fillna(""))
        flags = ok.turbine_design_range_flags.fillna("")
        stats["depaola_table2_flags"] = {k: int(flags.str.contains(f"turbine {k} ").sum())
                                         for k in ("an2_m2_rpm2", "loading_coefficient", "exit_mach", "exit_swirl_deg")}  # fmt: skip
        stats["sensitivity_feasible"] = {c[5:-9]: int(ok[c].astype(bool).sum()) for c in ok.columns
                                         if c.startswith("sens_") and c.endswith("_feasible")}  # fmt: skip
        stats["derived_ranges"] = {c: [round(float(ok[c].min()), 3), round(float(ok[c].max()), 3)]
                                   for c in ("derived_n_rpm", "derived_turb_phi", "derived_pr_c", "derived_comp_specific_speed_conventional") if c in ok}  # fmt: skip
    pa = ROOT / "validation" / "phase4" / f"pilot_analysis_{name}.yaml"
    pilot_analysis = yaml.safe_load(pa.read_text(encoding="utf-8")) if pa.exists() else None
    timing = sorted((ROOT / "runs").glob(f"*_dataset_{name}/timing.json"))
    if timing:
        t = json.loads(timing[-1].read_text(encoding="utf-8"))
        stats["last_run"] = t
    sdir = ROOT / "validation" / "phase4"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"dataset_stats_{name}.yaml").write_text(
        yaml.safe_dump(stats, sort_keys=False), encoding="utf-8"
    )

    pilot = n_done < n_total
    L = [f"# Dataset card: `{name}`{' (PILOT)' if pilot else ''}", "", f"> **{SAFETY_NOTE}**", "",
         f"Status: {n_done} of {n_total} planned designs evaluated" + (" (pilot: the first designs of the Sobol sequence)." if pilot else "."), ""]  # fmt: skip
    if pilot_analysis and pilot_analysis.get("stop"):
        L += [f"**STOPPED by the pilot stop rule:** {pilot_analysis['feasible']} of {pilot_analysis['pilot_designs']} pilot designs "
              f"feasible ({100 * pilot_analysis['feasible_fraction_of_pilot']:.1f} % < {100 * pilot_analysis['min_feasible_fraction']:.0f} %). "
              f"The full run was not started. Analysis: `validation/phase4/pilot_analysis_{name}.yaml`.", ""]  # fmt: skip
    mcfg = yaml.safe_load((ROOT / "configs" / "materials.yaml").read_text(encoding="utf-8"))
    life, sf = mcfg["design_life_h"]["value"], mcfg["stress_safety_factor"]["value"]
    at_ah, pf = (
        mcfg["blade_tip_to_hub_area_ratio"]["value"],
        mcfg["combustor_pattern_factor"]["value"],
    )
    L += [
         f"Generated with git {git} by `scripts/phase4/generate_dataset.py` (`turboml/data/generator.py`), config "
         f"`{Path(args.config).resolve().relative_to(ROOT)}`. Raw records: `datasets/<name>/designs.jsonl` (git-ignored, regenerate from seed + config).", "",
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
          f"inflated band (`{dcfg['calibration_sampling']}`); the others use the nominal values.", "",
          "Fixed assumptions: component losses `configs/engine_baseline.yaml` (DePaola values as summarised by the user, "
          "off-design loss scaling with corrected flow²); materials and creep settings `configs/materials.yaml` "
          f"(life {life:g} h, safety factor {sf}, blade taper At/Ah {at_ah} (K = {0.5 * (1 + at_ah):.2f}), pattern factor {pf}; "
          "NGV PF_max stored per design).", "",
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
    if "high_tit_split" in stats:
        L += ["High-TIT band (~15 % of the samples by design, D-056): feasibility inside and outside it:", "",
              "| band | designs | designed | feasible | % of designs | % of designed |", "|---|---|---|---|---|---|"]  # fmt: skip
        for k, v in stats["high_tit_split"].items():
            L.append(
                f"| {k} | {v['designs']} | {v['designed']} | {v['feasible']} | {v['feasible_%_of_designs']} | {v['feasible_%_of_designed']} |"
            )
        L += ["", "Infeasible high-TIT designs by active constraint (material status):", ""]
        L += md_table(
            stats["infeasible_high_tit_by_constraint"], "active constraints (material status)"
        ) + [""]
    if "high_tit_infeasibility_cause" in stats:
        c = stats["high_tit_infeasibility_cause"]
        L += ["Infeasible high-TIT designs: PHYSICS or MISSING DATA (`turboml/analysis/feasibility.py`; physics = every "
              "material fails a check its rupture data cover, or IN-713C metal above its solidus; missing data = some "
              "material fails only because a metal temperature is above its data):", ""]  # fmt: skip
        L += md_table(c["cause"], "cause") + [""]
        L += (
            ["Physics, IN-713C checks failed:", ""] + md_table(c["physics_in713c"], "checks") + [""]
        )
        L += (
            ["Missing data, data-limited material(s):", ""]
            + md_table(c["missing_data_limited_materials"], "materials")
            + [""]
        )
    if "tip_speed_regime" in stats:
        L += ["Compressor tip-speed regime (R3, D-060): **validated** = U2 <= 415 m/s (top measured Murray speed line), "
              "**extrapolated** = above it (wider calibration band, one-sided toward lower slip / higher loss):", "",
              "| regime / TIT band | designs | designed | feasible | % of designs | % of designed |", "|---|---|---|---|---|---|"]  # fmt: skip
        for k, v in stats["tip_speed_regime"].items():
            L.append(
                f"| {k} | {v['designs']} | {v['designed']} | {v['feasible']} | {v['feasible_%_of_designs']} | {v['feasible_%_of_designed']} |"
            )
        L.append("")
    if "turbine_eta_tt_design" in stats:
        t = stats["turbine_eta_tt_design"]
        L += ["Design-point turbine total-to-total efficiency (sanity check, D-069): quantiles "
              + ", ".join(f"{k} {v}" for k, v in t["quantiles"].items())
              + (f"; plausible range {t['plausible_range']} (Murray 2009 measured span +- published uncertainty); "
                 f"**{t.get('flagged', 0)} designs flagged** ({t.get('flagged_feasible', 0)} of them feasible), list in "
                 f"`validation/phase4/turbine_eta_flags_{name}.csv`." if t.get("plausible_range") else "."), ""]  # fmt: skip
    if "turbine_solver_retries" in stats:
        L += ["Turbine-solver restarts (D-067): points that converged only on a restart / failed after all restarts / "
              "recorded turbine_no_convergence:", "", "| point | converged on retry | failed after retries | turbine_no_convergence |", "|---|---|---|---|"]  # fmt: skip
        for pt, v in stats["turbine_solver_retries"].items():
            L.append(
                f"| {pt} | {v['converged_on_retry']} | {v['failed_after_retries']} | {v['turbine_no_convergence']} |"
            )
        L.append("")
    if "k_loss_subset" in stats:
        L += [
            "Turbine loss multiplier (D-064): nominal vs sampled subset:",
            "",
            "| subset | designs | designed | feasible | k_loss |",
            "|---|---|---|---|---|",
        ]
        for lab, v in stats["k_loss_subset"].items():
            L.append(
                f"| {lab} | {v['designs']} | {v['designed']} | {v['feasible']} | {v['k_loss_range'][0]} – {v['k_loss_range'][1]} |"
            )
        L.append("")
    if "t1_hub_tip" in stats:
        t = stats["t1_hub_tip"]
        L += [f"Turbine hub/tip (option T1, D-055): sampled {t['sampled_hub_tip_range'][0]} – {t['sampled_hub_tip_range'][1]}; "
              f"the exit-Mach-0.5 limit was the binding upper bound for {t['mach_limit_binding']} designed engines; "
              f"{t['no_buildable_hub_tip_designs']} designs had no buildable hub/tip (exit Mach > 0.5 even at 0.5).", ""]  # fmt: skip
    if "material_status" in stats:
        L += (
            ["Material status (designed engines):", ""]
            + md_table(stats["material_status"], "status")
            + [""]
        )
        L += [
            "Sensitivity (feasibility recomputed with one setting changed; the defaults above decide feasibility):",
            "",
        ]
        L += md_table(stats["sensitivity_feasible"], "setting", "feasible designs") + [""]
        L += [
            "Turbine outside DePaola 2020 Table 2 typical ranges (sanity flags only, not infeasible):",
            "",
        ]
        L += md_table(stats["depaola_table2_flags"], "quantity", "designs flagged") + [""]
        if stats["derived_ranges"]:
            L += ["Derived by the sizing (not sampled): " + "; ".join(f"{k.removeprefix('derived_')} {v[0]:g} – {v[1]:g}"
                                                                     for k, v in stats["derived_ranges"].items()), ""]  # fmt: skip
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
          "- Creep check uses typical (not minimum) rupture data; untapered blades (At/Ah = 1, conservative); MAR-M 247 data stop at 1311 K.",
          f"- Turbine geometry outside TurboFlow's recommended ranges (kept, warning in `geom_turb_sizing_warnings`): rotor "
          f"hub/tip in {stats['turbine_outside_turboflow_range']['rotor_hub_tip']} of {len(ok)} designed engines "
          f"(feasible among them: {stats['turbine_outside_turboflow_range']['feasible_with_rotor_hub_tip_warning']}); "
          "the loss correlations are less trustworthy there.",
          "- A compressor operating on its choke line can't be resolved (labelled `choke_compressor_*`).",
          "- Design variables are sampled independently in the box: many corners are physically impossible for this "
          "engine family (labelled design failures).", ""]  # fmt: skip
    cdir = ROOT / "docs" / "dataset_cards"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{name}.md").write_text("\n".join(L), encoding="utf-8")
    print(yaml.safe_dump(stats, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

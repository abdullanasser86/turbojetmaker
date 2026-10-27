"""Phase 5: train and evaluate the surrogates on one dataset version (one command; version-agnostic).

    uv run python scripts/phase5/train_surrogates.py --dataset phase4_v5          # GP + ensemble + classifier
    uv run python scripts/phase5/train_surrogates.py --dataset phase4_v6 --skip-gp

Pipeline (turboml/surrogates/, configs/surrogates.yaml):
1. data: inputs mapped to the sampler's unit cube (+ standardized calibration factors), targets transformed and
   standardized with TRAINING statistics; frozen test ids never used for fitting or tuning; seeded validation subset.
2. exact GP per target (fit on train), deep-ensemble regressor (train, early stopping on val), deep-ensemble
   feasibility classifier (train, early stopping on val).
3. metrics on val and on the frozen test set: accuracy (physical units), calibration (coverage, NLL), classifier
   accuracy/AUC/Brier/ECE; errors split by tip-speed regime (validated / extrapolated, D-060).
4. OOD detector fitted on the training inputs; flag rates and errors with / without flags on the test set.
Outputs: runs/<timestamp>_surrogates_<dataset>/ (models, config copy; git-ignored),
         validation/phase5/surrogates_<dataset>.yaml, docs/surrogate_cards/<dataset>.md.
Every number carries the dataset's result label (e.g. "provisional (pre-v6 calibration)" for phase4_v5).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from turboml.config import load_config
from turboml.device import get_device
from turboml.runlog import seed_everything, start_run
from turboml.surrogates import data as sdata
from turboml.surrogates import metrics
from turboml.surrogates.ensemble import EnsembleClassifier, EnsembleRegressor
from turboml.surrogates.gp import GPSurrogate
from turboml.surrogates.ood import OODDetector

ROOT = Path(__file__).resolve().parents[2]


def eval_regression(
    d: sdata.SurrogateData, sel: np.ndarray, mu_z: np.ndarray, sd_z: np.ndarray, levels
) -> dict:
    """Per-target accuracy (physical units) and calibration (standardized space) on the rows `sel`."""
    Z = d.Z()
    Y = d.Y_raw.to_numpy(float)
    Yhat = d.untransform(d.y_std.inverse(mu_z))
    out = {}
    for j, name in enumerate(d.target_names):
        out[name] = {"accuracy": metrics.regression(Y[sel, j], Yhat[sel, j]),
                     "calibration": metrics.calibration(Z[sel, j], mu_z[sel, j], sd_z[sel, j], levels)}  # fmt: skip
    return out


def by_regime(d, sel, mu_z, sd_z, levels) -> dict:
    out = {}
    for lab, m in (
        ("validated", d.tip_speed_m_s <= 415.0),
        ("extrapolated", d.tip_speed_m_s > 415.0),
    ):
        r = eval_regression(d, sel & m, mu_z, sd_z, levels)
        out[lab] = {k: {"rel_err_median": v["accuracy"].get("rel_err_median"), "r2": v["accuracy"].get("r2"),
                        "coverage_95": v["calibration"].get("coverage", {}).get("0.95")} for k, v in r.items()}  # fmt: skip
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--skip-gp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: tiny iteration counts (results meaningless)",
    )
    args = ap.parse_args()
    cfg = sdata.load_surrogate_config()
    if args.quick:
        cfg["gp"]["training_iters"] = 10
        cfg["ensemble"].update(epochs=10, members=2)
        cfg["classifier"].update(epochs=10, ensemble_size=2)
        cfg.setdefault("result_labels", {})[args.dataset] = "SMOKE TEST (--quick), meaningless"
    seed_everything(cfg["seed"])
    device = get_device(args.device)
    run_dir = start_run(load_config(), f"surrogates_{args.dataset}")
    shutil.copy(sdata.SURROGATE_CONFIG, run_dir / "surrogates_config.yaml")
    d = sdata.load(args.dataset, cfg)
    label = d.label or "final"
    print(f"dataset {args.dataset} [{label}] on {device}: {len(d.design_ids)} designs, "
          f"train {d.mask('train').sum()}, val {d.mask('val').sum()}, test {d.mask('test').sum()}")  # fmt: skip
    tr, va, te = d.mask("train"), d.mask("val"), d.mask("test")
    Z = d.Z()
    levels = cfg["metrics"]["interval_levels"]
    res = {"dataset": args.dataset, "label": label, "device": str(device), "run_dir": str(run_dir.relative_to(ROOT)),
           "inputs": d.input_names, "targets": d.target_names,
           "counts": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                      "designed_train": int((tr & np.isfinite(Z[:, 0])).sum())}, "models": {}}  # fmt: skip
    timing = {}

    # --- exact GP
    if not args.skip_gp:
        t0 = time.perf_counter()
        gp = GPSurrogate(cfg["gp"], device=device, seed=cfg["seed"]).fit(d.U[tr], Z[tr])
        timing["gp_fit_s"] = time.perf_counter() - t0
        gp.save(run_dir / "models" / "gp.pt")
        mu, sd = gp.predict(d.U)
        res["models"]["gp"] = {"val": eval_regression(d, va, mu, sd, levels), "test": eval_regression(d, te, mu, sd, levels),
                               "test_by_tip_speed_regime": by_regime(d, te, mu, sd, levels)}  # fmt: skip
    # --- deep ensemble regressor
    t0 = time.perf_counter()
    ens = EnsembleRegressor(
        cfg["ensemble"], device=device, seed=cfg["seed"], ckpt_dir=run_dir / "ckpt" / "ens"
    )
    ens.fit(d.U[tr], Z[tr], d.U[va], Z[va])
    timing["ensemble_fit_s"] = time.perf_counter() - t0
    ens.save(run_dir / "models" / "ensemble.pt")
    mu, sd, epi = ens.predict(d.U)
    res["models"]["ensemble"] = {"val": eval_regression(d, va, mu, sd, levels), "test": eval_regression(d, te, mu, sd, levels),
                                 "test_by_tip_speed_regime": by_regime(d, te, mu, sd, levels)}  # fmt: skip
    # --- feasibility classifier (all designs)
    t0 = time.perf_counter()
    clf = EnsembleClassifier(
        cfg["classifier"], device=device, seed=cfg["seed"] + 1, ckpt_dir=run_dir / "ckpt" / "clf"
    )
    clf.fit(d.U[tr], d.feasible[tr], d.U[va], d.feasible[va])
    timing["classifier_fit_s"] = time.perf_counter() - t0
    clf.save(run_dir / "models" / "classifier.pt")
    p, p_sd = clf.predict_proba(d.U)
    res["models"]["feasibility_classifier"] = {
        "val": metrics.classification(d.feasible[va], p[va], cfg["metrics"]["ece_bins"]),
        "test": metrics.classification(d.feasible[te], p[te], cfg["metrics"]["ece_bins"]),
        "test_by_tip_speed_regime": {lab: metrics.classification(d.feasible[te & m], p[te & m], cfg["metrics"]["ece_bins"])
                                     for lab, m in (("validated", d.tip_speed_m_s <= 415.0), ("extrapolated", d.tip_speed_m_s > 415.0))}}  # fmt: skip
    # --- OOD flags (fitted on training inputs and training epistemic spread)
    designed_tr = tr & np.isfinite(Z[:, 0])
    ood = OODDetector(cfg["ood"], d.n_design_vars).fit(d.U[designed_tr], epi[designed_tr])
    fl = ood.flags(d.U, d.tip_speed_m_s, epi)
    sel = te & np.isfinite(Z[:, 0])
    res["ood"] = {"thresholds": {"knn_mean_distance": ood.knn_threshold},
                  "test_flag_rate": {k: round(float(v[sel].mean()), 3) for k, v in fl.items()}}  # fmt: skip
    for lab, m in (("flagged", fl["any"]), ("not_flagged", ~fl["any"])):
        r = eval_regression(d, sel & m, mu, sd, levels)
        res["ood"][f"ensemble_test_{lab}"] = {k: {"rel_err_median": v["accuracy"].get("rel_err_median"),
                                                  "coverage_95": v["calibration"].get("coverage", {}).get("0.95"),
                                                  "n": v["accuracy"].get("n")} for k, v in r.items()}  # fmt: skip
    res["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    out = ROOT / "validation" / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (out / f"surrogates_{args.dataset}{'_quick' if args.quick else ''}.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    if not args.quick:
        write_card(res)
    print(yaml.safe_dump({k: res[k] for k in ("label", "counts", "timing_s")}, sort_keys=False))
    return 0


def write_card(res: dict) -> None:
    lab = res["label"]
    L = [f"# Surrogate card: `{res['dataset']}` — **{lab}**", "",
         f"Generated by `scripts/phase5/train_surrogates.py --dataset {res['dataset']}` on {res['device']}; models in "
         f"`{res['run_dir']}` (git-ignored); full metrics `validation/phase5/surrogates_{res['dataset']}.yaml`.", "",
         f"**Every number below is {lab}.**" if lab != "final" else "", "",
         f"Inputs: {', '.join(res['inputs'])}. Counts: train {res['counts']['train']}, val {res['counts']['val']}, "
         f"frozen test {res['counts']['test']} (designed engines in train: {res['counts']['designed_train']}).", "",
         "## Regression, frozen test set (physical units; coverage = share of points inside the central 95 % interval)", "",
         "| target | model | n | median rel. err | p95 rel. err | R² | 95 % coverage | std(z) |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for model in ("gp", "ensemble"):
        if model not in res["models"]:
            continue
        for t, v in res["models"][model]["test"].items():
            a, c = v["accuracy"], v["calibration"]

            def f(x, pct=False):
                return "–" if x is None else (f"{100 * x:.2f} %" if pct else f"{x:.3f}")

            L.append(f"| {t} | {model} | {a.get('n')} | {f(a.get('rel_err_median'), True)} | {f(a.get('rel_err_p95'), True)} | "
                     f"{f(a.get('r2'))} | {f(c.get('coverage', {}).get('0.95'))} | {f(c.get('std_of_standardized_residual'))} |")  # fmt: skip
    c = res["models"]["feasibility_classifier"]["test"]
    L += ["", "## Feasibility classifier, frozen test set", "",
          f"accuracy {c['accuracy']:.3f}, ROC AUC {c['roc_auc']:.3f}, Brier {c['brier']:.3f}, ECE {c['ece']:.3f} "
          f"(n {c['n']}, feasible {c['positives']}).", "",
          "## Out-of-distribution flags (test set, designed engines)", ""]  # fmt: skip
    L += [f"- {k}: {100 * v:.1f} %" for k, v in res["ood"]["test_flag_rate"].items()]
    L += ["", "Flags: outside the sampled box; extrapolated compressor tip speed (> 415 m/s, D-060); sparse region (kNN "
          "distance); high ensemble disagreement. A flagged design must be re-evaluated with the physics model.", "",
          "Limitations: trained on the 0-D physics model's outputs (not on engine tests); inherits every model-form "
          "limit of the dataset (see its dataset card); final results are always physics-verified."]  # fmt: skip
    p = ROOT / "docs" / "surrogate_cards" / f"{res['dataset']}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

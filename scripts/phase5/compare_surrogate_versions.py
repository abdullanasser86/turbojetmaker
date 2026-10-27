"""Compare surrogate metrics of two dataset versions on their common targets (frozen test sets of each version).

    uv run python scripts/phase5/compare_surrogate_versions.py phase4_v5 phase4_v6

Reads validation/phase5/surrogates_<a>.yaml and _<b>.yaml (written by train_surrogates.py) and writes
validation/phase5/compare_<a>_vs_<b>.md: per target and model the median / p95 relative error, R^2, 95 % coverage and
std of standardized residuals, plus the feasibility-classifier metrics. Each version keeps its own result label.
Note: the test sets differ (each version has its own frozen split and its own physics calibration), so this compares
how well each surrogate reproduces ITS dataset, not the physics of the two versions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.2f} %" if pct else f"{x:.3f}"


def main(a: str, b: str) -> int:
    ra, rb = (yaml.safe_load((ROOT / "validation" / "phase5" / f"surrogates_{n}.yaml").read_text(encoding="utf-8")) for n in (a, b))
    L = [f"# Surrogates: `{a}` vs `{b}` (frozen test set of each version)", "",
         f"- `{a}`: {ra['label']}", f"- `{b}`: {rb['label']}", "",
         "Test sets differ (own frozen split and own calibration per version): this compares how well each surrogate "
         "reproduces its own dataset.", "",
         "| target | model | median rel. err | p95 rel. err | R² | 95 % coverage | std(z) |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for model in ("gp", "ensemble"):
        if model not in ra["models"] or model not in rb["models"]:
            continue
        ta, tb = ra["models"][model]["test"], rb["models"][model]["test"]
        for t in [t for t in tb if t in ta] + [t for t in tb if t not in ta]:
            cells = []
            for key, sub, pct in (("accuracy", "rel_err_median", True), ("accuracy", "rel_err_p95", True), ("accuracy", "r2", False)):
                va = ta.get(t, {}).get(key, {}).get(sub)
                vb = tb[t][key].get(sub)
                cells.append(f"{fmt(va, pct)} → {fmt(vb, pct)}")
            ca = ta.get(t, {}).get("calibration", {})
            cb = tb[t]["calibration"]
            cells.append(f"{fmt(ca.get('coverage', {}).get('0.95'))} → {fmt(cb.get('coverage', {}).get('0.95'))}")
            cells.append(f"{fmt(ca.get('std_of_standardized_residual'))} → {fmt(cb.get('std_of_standardized_residual'))}")
            L.append(f"| {t}{'' if t in ta else ' (new)'} | {model} | " + " | ".join(cells) + " |")
    ca, cb = ra["models"]["feasibility_classifier"]["test"], rb["models"]["feasibility_classifier"]["test"]
    L += ["", "| classifier | accuracy | ROC AUC | Brier | ECE | n (feasible) |", "|---|---|---|---|---|---|"]
    for n, c in ((a, ca), (b, cb)):
        L.append(f"| {n} | {c['accuracy']:.3f} | {c['roc_auc']:.3f} | {c['brier']:.3f} | {c['ece']:.3f} | {c['n']} ({c['positives']}) |")
    out = ROOT / "validation" / "phase5" / f"compare_{a}_vs_{b}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

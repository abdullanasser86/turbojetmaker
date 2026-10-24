"""Phase 1 acceptance: pyCycle vs independent 0-D cycle over a Sobol sweep.

Both models get identical inputs; the 0-D model additionally receives pyCycle's ambient
static T and p so that the two US-1976 implementations (table vs formula) do not enter
the comparison (they are compared separately in tests/test_atmosphere.py).

Usage:  uv run python scripts/phase1/cross_check_sweep.py
Output: validation/phase1/cross_check.csv and a summary table on stdout.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

from turboml.physics.cycle_common import CycleInputs
from turboml.physics.cycle_pycycle import PyCycleDesignModel
from turboml.physics.cycle_simple import design_point

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "phase1_crosscheck.yaml").read_text())
OUT = ROOT / "validation" / "phase1" / "cross_check.csv"

COMPARE = ["Fn_N", "TSFC_kg_N_s", "far", "pr_t", "P_comp_W", "EGT_K", "A9_m2", "V9_m_s"]


def samples() -> list[CycleInputs]:
    names = list(CFG["ranges"])
    lo = np.array([CFG["ranges"][k][0] for k in names])
    hi = np.array([CFG["ranges"][k][1] for k in names])
    u = qmc.Sobol(d=len(names), scramble=True, seed=CFG["seed"]).random(CFG["n_samples"])
    X = qmc.scale(u, lo, hi)
    return [CycleInputs(**dict(zip(names, row, strict=True)), **CFG["fixed"]) for row in X]


def main() -> int:
    xs = samples()
    rows = []
    for thermo in CFG["thermo"]:
        model = PyCycleDesignModel(thermo)
        t0 = time.time()
        for i, x in enumerate(xs):
            rp = model.evaluate(x)
            Ts, Ps = model.ambient() if rp.converged else (None, None)
            rs = design_point(replace(x, Ts0_K=Ts, Ps0_Pa=Ps))
            row = {"sample": i, "thermo": thermo, **{k: getattr(x, k) for k in CFG["ranges"]}}
            row["pyc_converged"], row["pyc_failure"] = rp.converged, rp.failure_reason
            row["simple_converged"], row["simple_failure"] = rs.converged, rs.failure_reason
            row["choked_pyc"], row["choked_simple"] = rp.choked, rs.choked
            for k in COMPARE:
                a, b = getattr(rp, k), getattr(rs, k)
                row[f"{k}_pyc"], row[f"{k}_simple"] = a, b
                row[f"{k}_reldiff"] = (a - b) / b if rp.converged and rs.converged else np.nan
            rows.append(row)
        print(f"{thermo}: {len(xs)} points in {time.time() - t0:.1f} s")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    tol = CFG["tolerance_rel"]
    print(f"\nRelative difference pyCycle vs 0-D (tolerance {tol:.0%} on Fn and TSFC)")
    ok_all = True
    for thermo, g in df.groupby("thermo"):
        both = g.pyc_converged & g.simple_converged
        print(
            f"\n[{thermo}] converged: pyCycle {g.pyc_converged.sum()}/{len(g)}, "
            f"0-D {g.simple_converged.sum()}/{len(g)}; choked-flag agreement "
            f"{(g.choked_pyc == g.choked_simple)[both].mean():.0%}; "
            f"choked points {g.choked_simple[both].sum()}"
        )
        for k in COMPARE:
            d = g.loc[both, f"{k}_reldiff"].abs()
            print(f"  {k:12s} median {d.median():.3%}  max {d.max():.3%}")
        for k in ("Fn_N", "TSFC_kg_N_s"):
            ok_all &= bool((g.loc[both, f"{k}_reldiff"].abs() < tol).all()) and bool(both.all())
    print("\nACCEPTANCE (all points converge, |dFn|,|dTSFC| < tol):", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

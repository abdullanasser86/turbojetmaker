"""Phase 1 acceptance: pyCycle-CEA vs independent 0-D cycle over a Sobol sweep.

Uses the same cross-check machinery that later guards every generated dataset
(:mod:`turboml.physics.crosscheck`).

PRIMARY metric (acceptance): |ΔFn/Fn| and |ΔTSFC/TSFC| < 1 %.
SECONDARY metric (reported, labelled, never used to pass/fail): |ΔFn|/Fg.
Known exception: sample 8 (validation/phase1/known_exceptions.yaml).

Usage:  uv run python scripts/phase1/cross_check_sweep.py
Output: validation/phase1/cross_check.csv and a summary on stdout.
History: the Phase 1 run (2026-10-24) also compared pyCycle-TABULAR (now dropped); its
numbers are recorded in docs/VALIDATION.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

from turboml.physics import crosscheck
from turboml.physics.cycle_common import CycleInputs

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "phase1_crosscheck.yaml").read_text(encoding="utf-8"))
EXC = yaml.safe_load(
    (ROOT / "validation" / "phase1" / "known_exceptions.yaml").read_text(encoding="utf-8")
)
OUT = ROOT / "validation" / "phase1" / "cross_check.csv"


def samples() -> list[CycleInputs]:
    names = list(CFG["ranges"])
    lo = np.array([CFG["ranges"][k][0] for k in names])
    hi = np.array([CFG["ranges"][k][1] for k in names])
    u = qmc.Sobol(d=len(names), scramble=True, seed=CFG["seed"]).random(CFG["n_samples"])
    X = qmc.scale(u, lo, hi)
    return [CycleInputs(**dict(zip(names, row, strict=True)), **CFG["fixed"]) for row in X]


def main() -> int:
    xs = samples()
    t0 = time.time()
    df = crosscheck.run(xs)
    inputs = pd.DataFrame([{k: getattr(x, k) for k in CFG["ranges"]} for x in xs])
    df = pd.concat([inputs, df.drop(columns="index")], axis=1)
    df.insert(0, "sample", range(len(df)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"pyCycle-CEA vs 0-D: {len(xs)} points in {time.time() - t0:.1f} s")

    known = {e["sample"] for e in EXC["exceptions"]}
    print("\nRelative differences (PRIMARY metrics; SECONDARY row labelled):")
    print(crosscheck.summary(df).map(lambda v: f"{v:.3%}").to_string())
    flagged = df[df.flagged]
    print(f"\nflagged (|dFn/Fn| or |dTSFC/TSFC| > {crosscheck.FLAG_TOL_REL:.0%}): "
          f"{list(flagged['sample'])}  (known exceptions: {sorted(known)})")  # fmt: skip
    print(crosscheck.gate(df))
    unexplained = set(flagged["sample"]) - known
    ok = not unexplained
    print(
        "ACCEPTANCE (all flagged points are documented known exceptions):", "PASS" if ok else "FAIL"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

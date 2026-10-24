"""Cross-check of the 0-D cycle model (the data generator) against pyCycle-CEA.

Policy (user decision, 2026-10-24; docs/DECISIONS.md D-016):
* Training data come from the fast 0-D model (`cycle_simple`).
* pyCycle-CEA is re-run in parallel on a random 2 % of every generated dataset and on
  every final Pareto design.
* A point is *flagged* when |ΔFn/Fn| or |ΔTSFC/TSFC| > 1 % (PRIMARY metric).
* If more than 5 % of the cross-checked points are flagged, the pipeline must stop and
  report before training surrogates (:func:`gate`).
* The gross-thrust-relative difference |ΔFn|/Fg is also reported as a SECONDARY metric,
  labelled as such. It is never used to pass or fail. It exists because Fn/Fn is
  ill-conditioned when net thrust is a small difference of gross thrust and ram drag
  (see docs/VALIDATION.md, "known exception").

Both models get identical inputs. The 0-D model additionally receives pyCycle's ambient
static T and p, so that the atmosphere implementations don't enter the comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from turboml.physics.cycle_common import CycleInputs
from turboml.physics.cycle_simple import design_point

FLAG_TOL_REL = 0.01  # primary metric tolerance on Fn and TSFC
GATE_MAX_FLAGGED_FRAC = 0.05  # stop if more than this fraction is flagged
SAMPLE_FRACTION = 0.02

COMPARED = ("Fn_N", "Fg_N", "TSFC_kg_N_s", "far", "pr_t", "P_comp_W", "EGT_K", "V9_m_s", "A9_m2")


def _check_chunk(items: Sequence[tuple[int, CycleInputs]]) -> list[dict]:
    from turboml.physics.cycle_pycycle import PyCycleDesignModel  # heavy import, per worker

    model = PyCycleDesignModel()
    rows = []
    for idx, x in items:
        rp = model.evaluate(x)
        Ts, Ps = model.ambient() if rp.converged else (None, None)
        rs = design_point(replace(x, Ts0_K=Ts, Ps0_Pa=Ps))
        row = {"index": idx, "pyc_converged": rp.converged, "pyc_failure": rp.failure_reason,
               "simple_converged": rs.converged, "simple_failure": rs.failure_reason,
               "choked_pyc": rp.choked, "choked_simple": rs.choked}  # fmt: skip
        both = rp.converged and rs.converged
        for k in COMPARED:
            a, b = getattr(rp, k), getattr(rs, k) if rs.converged else np.nan
            row[f"{k}_pyc"], row[f"{k}_simple"] = a, b
            row[f"{k}_reldiff"] = (a - b) / b if both else np.nan
        # SECONDARY metric (labelled): net-thrust difference relative to gross thrust
        row["secondary_dFn_over_Fg"] = (rp.Fn_N - rs.Fn_N) / rs.Fg_N if both else np.nan
        rows.append(row)
    return rows


def run(
    inputs: Sequence[CycleInputs], indices: Sequence[int] | None = None, n_jobs: int = -1
) -> pd.DataFrame:
    """Cross-check the given designs in parallel (one pyCycle model per worker)."""
    indices = list(range(len(inputs))) if indices is None else list(indices)
    items = list(zip(indices, inputs, strict=True))
    n_workers = max(1, min(len(items), (n_jobs if n_jobs > 0 else _cpu_count())))
    chunks = [items[i::n_workers] for i in range(n_workers)]
    results = Parallel(n_jobs=n_workers)(delayed(_check_chunk)(c) for c in chunks if c)
    df = pd.DataFrame([r for chunk in results for r in chunk]).sort_values("index")
    both = df.pyc_converged & df.simple_converged
    df["flagged"] = (
        ~both
        | (df.Fn_N_reldiff.abs() > FLAG_TOL_REL)
        | (df.TSFC_kg_N_s_reldiff.abs() > FLAG_TOL_REL)
    )
    return df.reset_index(drop=True)


def sample_indices(
    n_total: int, rng: np.random.Generator, fraction: float = SAMPLE_FRACTION
) -> np.ndarray:
    """Random subset (at least one point) of a dataset of size n_total, for cross-checking."""
    k = max(1, int(np.ceil(fraction * n_total)))
    return np.sort(rng.choice(n_total, size=k, replace=False))


@dataclass
class GateResult:
    n_checked: int
    n_flagged: int
    flagged_frac: float
    passed: bool

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "STOP - report before training surrogates"
        return (f"cross-check: {self.n_flagged}/{self.n_checked} flagged "
                f"({self.flagged_frac:.1%}, limit {GATE_MAX_FLAGGED_FRAC:.0%}) -> {verdict}")  # fmt: skip


def gate(df: pd.DataFrame, max_flagged_frac: float = GATE_MAX_FLAGGED_FRAC) -> GateResult:
    n, f = len(df), int(df.flagged.sum())
    frac = f / n if n else 1.0
    return GateResult(n_checked=n, n_flagged=f, flagged_frac=frac, passed=frac <= max_flagged_frac)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Median / max absolute relative differences (primary) and the secondary metric."""
    both = df.pyc_converged & df.simple_converged
    rows = {
        k: (df.loc[both, f"{k}_reldiff"].abs().median(), df.loc[both, f"{k}_reldiff"].abs().max())
        for k in COMPARED
    }
    rows["SECONDARY |dFn|/Fg"] = (
        df.loc[both, "secondary_dFn_over_Fg"].abs().median(),
        df.loc[both, "secondary_dFn_over_Fg"].abs().max(),
    )
    return pd.DataFrame(rows, index=["median_abs", "max_abs"]).T


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1

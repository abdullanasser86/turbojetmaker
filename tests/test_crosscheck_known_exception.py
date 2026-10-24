"""Known exception of the pyCycle-CEA vs 0-D cross-check (validation/phase1/known_exceptions.yaml).

The point is ill-conditioned (nozzle pressure ratio 1.09, Fg/Fn = 5.6), so its primary-metric
difference is ~1 %. This test pins that behaviour: a change in either model that moves it
outside the documented band must be investigated, not silently accepted.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from turboml.physics import crosscheck
from turboml.physics.cycle_common import CycleInputs

ROOT = Path(__file__).resolve().parents[1]
EXC = yaml.safe_load((ROOT / "validation" / "phase1" / "known_exceptions.yaml").read_text())


@pytest.mark.slow
@pytest.mark.parametrize("exc", EXC["exceptions"], ids=lambda e: f"sample{e['sample']}")
def test_known_exception_stays_explained(exc):
    df = crosscheck.run([CycleInputs(**exc["inputs"])], n_jobs=1)
    row = df.iloc[0]
    lo, hi = exc["regression_bounds"]["dFn_over_Fn"]
    assert lo <= row.Fn_N_reldiff <= hi
    assert (
        abs(row.secondary_dFn_over_Fg) < exc["regression_bounds"]["abs_secondary_dFn_over_Fg_max"]
    )
    assert bool(row.flagged)  # still counted against the 5 % gate; it is not hidden


def test_gate_logic():
    import pandas as pd

    df = pd.DataFrame({"flagged": [False] * 95 + [True] * 5})
    assert crosscheck.gate(df).passed  # exactly 5 % passes
    df = pd.DataFrame({"flagged": [False] * 94 + [True] * 6})
    assert not crosscheck.gate(df).passed


def test_sample_indices_two_percent():
    idx = crosscheck.sample_indices(5000, np.random.default_rng(0))
    assert len(idx) == 100 and len(set(idx)) == 100 and idx.max() < 5000
    assert len(crosscheck.sample_indices(10, np.random.default_rng(0))) == 1

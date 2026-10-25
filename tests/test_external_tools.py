"""Smoke + regression tests for the external physics codes in the pinned environment.

These are REGRESSION tests (values recorded from our own Phase 0 runs on 2026-10-24), not
validation: they detect when a dependency upgrade changes results. Validation against
measured data is tracked in docs/VALIDATION.md.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "third_party"


@pytest.mark.slow
def test_pycycle_simple_turbojet_example(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(THIRD / "pycycle_examples" / "simple_turbojet.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.splitlines()
    summaries = [lines[i + 1].split() for i, ln in enumerate(lines) if "Mach      Alt" in ln]
    assert len(summaries) == 3  # DESIGN, OD0, OD1
    # columns: Mach Alt W Fn Fg Fram OPR TSFC   (US units: lbm/s, lbf, lbm/(lbf h))
    fn = [float(s[3]) for s in summaries]
    tsfc = [float(s[7]) for s in summaries]
    np.testing.assert_allclose(fn, [11800.0, 11000.0, 8000.0], rtol=1e-4)
    np.testing.assert_allclose(tsfc, [0.79852, 0.78364, 0.82951], rtol=1e-4)


@pytest.mark.slow
def test_radcomp_notebook_case():
    import yaml
    from radcompressor.compressor import Compressor
    from radcompressor.condition import OperatingCondition
    from radcompressor.geometry import Geometry
    from radcompressor.thermo import CoolPropFluid

    case = yaml.safe_load((THIRD / "radcomp_known_compressors.yml").read_text(encoding="utf-8"))[-1]
    c = case["conditions"]
    in0 = CoolPropFluid(c["fluid"]).thermo_prop("PT", c["in_P"], c["in_T"])
    op = OperatingCondition(in0=in0, fld=in0.fld, m=0.120, n_rot=130e3 * np.pi / 30)
    comp = Compressor(Geometry.from_dict(case["geom"]), op)
    assert comp.calculate()
    assert comp.PR == pytest.approx(1.4976, rel=1e-3)
    assert comp.eff == pytest.approx(0.6953, rel=1e-3)


@pytest.mark.slow
def test_turboflow_one_stage_example():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import turboflow as tf

    config = tf.load_config(str(THIRD / "turboflow_one_stage_config.yaml"), print_summary=False)
    solvers = tf.compute_performance(
        config["operation_points"], config, export_results=False, stop_on_failure=True
    )
    assert solvers[0].success
    r = solvers[0].problem.results["overall"]
    assert float(r["efficiency_ts"].iloc[0]) == pytest.approx(79.12, abs=0.02)  # percent
    assert float(r["mass_flow_rate"].iloc[0]) == pytest.approx(2.7371, rel=1e-3)

"""Phase 0: TurboFlow hello-world.

Runs TurboFlow's `one-stage_turbine` example (vendored config, unmodified) in
performance-analysis mode at its single operating point. The geometry is the NASA
single-stage cold-air turbine of Kofskey et al. (1972), for which TurboFlow ships
experimental data in its `validation_cases/` -- a candidate validation case for Phase 2.

Note: TurboFlow reports efficiencies in percent.

Usage:  uv run python scripts/phase0/run_turboflow_hello.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import turboflow as tf  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "third_party" / "turboflow_one_stage_config.yaml"


def main() -> int:
    config = tf.load_config(str(CONFIG), print_summary=False)
    t0 = time.time()
    solvers = tf.compute_performance(
        config["operation_points"], config, export_results=False, stop_on_failure=True
    )
    r = solvers[0].problem.results["overall"]
    get = lambda k: float(r[k].iloc[0])  # noqa: E731
    print(
        f"\nTurboFlow one-stage (Kofskey 1972 geometry, air): converged={solvers[0].success}\n"
        f"  eta_ts = {get('efficiency_ts'):.2f} %   eta_tt = {get('efficiency_tt'):.2f} %\n"
        f"  mdot = {get('mass_flow_rate'):.4f} kg/s   PR_ts = {get('PR_ts'):.3f}   "
        f"power = {get('power') / 1e3:.2f} kW   wall time {time.time() - t0:.1f} s"
    )
    return 0 if solvers[0].success else 1


if __name__ == "__main__":
    raise SystemExit(main())

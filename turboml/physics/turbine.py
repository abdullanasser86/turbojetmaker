"""Axial turbine meanline model (wrapper around TurboFlow's axial-turbine solver).

What a meanline axial turbine model does
----------------------------------------
Hot gas from the combustor is turned by a ring of stationary vanes (the *stator* or nozzle
guide vanes, NGV), which accelerates it and gives it swirl. A ring of rotating blades (the
*rotor*) then takes that swirl out, extracting work (Euler: w = U*(C_theta,in - C_theta,out)).
Along a mean-radius streamline the model solves continuity, energy and the velocity
triangles in each blade row. Loss correlations (here the Benner et al. set, profile +
secondary + tip-clearance + trailing-edge losses) give the entropy rise, a deviation model
gives the exit flow angle, and a choking criterion limits the flow when a throat reaches
Mach 1. Output: mass flow, efficiency, power/torque and exit flow angle at a given pressure
ratio and speed. This is what the engine matching solver (Phase 3) needs.

Implementation and scope
------------------------
* Solver: TurboFlow 0.1.18 axial turbine (Anderson et al., JOSS 10(111) 7588, 2025),
  equation-oriented, validated by its authors against Kofskey's NASA cold-air turbine tests.
  The JOSS paper validates this axial-turbine part (unlike its centrifugal-compressor module,
  see docs/DECISIONS.md D-021).
* Working fluid: TurboFlow evaluates properties with CoolProp. Combustion products are not a
  CoolProp pure fluid, so we run it with "air". The resulting error is quantified in
  scripts/phase2/turbine_gas_error.py (see docs/VALIDATION.md).
* ~3 s per operating point. Fine for maps; too slow for millions of evaluations (Phase 3 will
  tabulate maps per geometry or use a surrogate).
* Failures (non-convergence) are returned as rows with ``converged=False``, never dropped.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KOFSKEY_1972_CONFIG = ROOT / "third_party" / "turboflow_kofskey1972_1stage.yaml"


class AxialTurbine:
    """TurboFlow axial turbine with TurboML conventions (SI, failures as data)."""

    def __init__(self, config_path: str | Path = KOFSKEY_1972_CONFIG):
        os.environ.setdefault("MPLBACKEND", "Agg")
        import turboflow as tf

        self._tf = tf
        with contextlib.redirect_stdout(io.StringIO()):
            self.config = tf.load_config(str(config_path), print_summary=False)
        self.design_point = dict(self.config["operation_points"])

    @classmethod
    def from_geometry(cls, geometry: dict, T0_in_K: float, p0_in_Pa: float, omega_rad_s: float,
                      template: str | Path = KOFSKEY_1972_CONFIG, fluid_name: str = "air") -> AxialTurbine:  # fmt: skip
        """TurboFlow model for any stage geometry (TurboFlow geometry dict, SI/degrees), using the
        solver/loss settings of `template`. The design point (p_out) is set to p0/2 as a placeholder;
        use `operation_point` to choose the evaluated point."""
        import tempfile

        import yaml

        cfg = yaml.safe_load(Path(template).read_text())
        cfg["geometry"] = {
            k: [float(x) for x in v] if k != "cascade_type" else list(v)
            for k, v in geometry.items()
        }
        op = {"fluid_name": fluid_name, "T0_in": float(T0_in_K), "p0_in": float(p0_in_Pa),
              "p_out": float(p0_in_Pa) / 2.0, "omega": float(omega_rad_s), "alpha_in": 0}  # fmt: skip
        cfg["operation_points"] = op
        cfg["performance_analysis"]["performance_map"] = dict(op)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            path = f.name
        try:
            return cls(path)
        finally:
            os.unlink(path)

    def operation_point(self, pr_ts: float, speed_frac: float = 1.0, **overrides) -> dict:
        """Operating point at total-to-static pressure ratio p0_in/p_out and fraction of design speed."""
        op = copy.deepcopy(self.design_point)
        op.update(overrides)
        op["p_out"] = op["p0_in"] / pr_ts
        op["omega"] = self.design_point["omega"] * speed_frac
        return op

    def evaluate(self, points: list[dict]) -> pd.DataFrame:
        """Evaluate a list of operating points (TurboFlow continuation between neighbours)."""
        error = ""
        try:
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
                solvers = self._tf.compute_performance(
                    points, self.config, export_results=False, stop_on_failure=False
                )
        except Exception as e:  # noqa: BLE001 -- TurboFlow can crash in its own summary printout
            solvers, error = [None] * len(points), f"turboflow_error: {type(e).__name__}: {e}"[:150]
        rows = []
        for op, s in zip(points, solvers, strict=True):
            row = {"p0_in_Pa": op["p0_in"], "T0_in_K": op["T0_in"], "pr_ts": op["p0_in"] / op["p_out"],
                   "omega_rad_s": op["omega"], "speed_frac": op["omega"] / self.design_point["omega"]}  # fmt: skip
            if s is None or not getattr(s, "success", False):
                rows.append({**row, "converged": False, "failure_reason": error or "no_convergence"})
                continue
            r = s.problem.results
            o = r["overall"]
            g = lambda k, o=o: float(np.asarray(o[k]).ravel()[0])  # noqa: E731
            plane = r["plane"]
            rows.append({
                **row, "converged": True, "failure_reason": "",
                "mass_flow_kg_s": g("mass_flow_rate"), "eta_ts": g("efficiency_ts") / 100.0,
                "eta_tt": g("efficiency_tt") / 100.0, "power_W": g("power"), "torque_N_m": g("torque"),
                "exit_flow_angle_deg": float(np.asarray(plane["alpha"]).ravel()[-1]),
            })  # fmt: skip
        return pd.DataFrame(rows)

    def map(self, speed_fracs, pr_ts_values) -> pd.DataFrame:
        """Turbine map: each speed line evaluated with increasing pressure ratio (continuation)."""
        pts = [self.operation_point(pr, sf) for sf in speed_fracs for pr in pr_ts_values]
        return self.evaluate(pts)

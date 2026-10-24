"""TurboFlow centrifugal-compressor wrapper -- CROSS-CHECK ONLY (see docs/DECISIONS.md D-021).

The primary compressor model is :mod:`turboml.physics.compressor`. This wrapper runs
TurboFlow 0.1.18 (impeller with the Oh/Zhang loss sets) with our Wiesner slip fix. Its
vaneless diffuser is lossless (D-021), so treat its stage results as impeller-only.

Original notes follow.


What a meanline compressor model does
-------------------------------------
A centrifugal compressor sucks air in axially at the *inducer* (eye), and the spinning
*impeller* blades fling it outward. That raises its pressure (centrifugal effect) and its
kinetic energy. A *diffuser* outside the impeller then slows the air down and turns that
kinetic energy into more pressure.

A meanline model follows one representative streamline. At inlet and exit it builds the
**velocity triangles**: the blade speed U = omega*r, the absolute flow velocity C and the
velocity relative to the blade W = C - U. The **Euler turbine equation** gives the work put
into each kg of air:

    w_Euler = U2 * C_theta2 - U1 * C_theta1          (C_theta: swirl velocity)

The ideal swirl at the exit would follow the blade angle, but the flow "slips" behind it.
The *slip factor* sigma corrects that: C_theta2 = sigma*U2 - C_m2*tan(beta2_blade), with
sigma ~ 1 - sqrt(cos beta2)/Z^0.7 (Wiesner 1967; Z = blade count). Empirical **loss
correlations** then remove part of that work as entropy (incidence, blade loading, skin
friction, tip clearance, wake mixing, plus parasitic disk friction, recirculation and
leakage). That yields the total pressure ratio and isentropic efficiency.

Implementation
--------------
The equations are solved by TurboFlow 0.1.18 (Anderson et al., JOSS 10(111) 7588, 2025),
which implements the loss sets of Oh, Yoon & Chung (1997) ("oh") and Zhang et al.
("zhang_set1..3"). This module adds:

* geometry from YAML with per-field provenance;
* **corrected Wiesner slip**: TurboFlow 0.1.18 evaluates cos() of the trailing-edge angle in
  radians although the angle is in degrees (wrong slip; NaN for |beta2| > ~33 deg). Our hook uses
  degrees (docs/DECISIONS.md D-021);
* **slip-factor uncertainty**: sigma -> sigma + delta_sigma. Harrison (Purdue PhD 2020, as quoted
  in PROJECT_BRIEF) found Wiesner within +/-0.05 for only about half of the impellers tested,
  so delta_sigma is treated as an uncertain input, not a constant;
* robust speed lines: continuation in mass flow, every failure recorded with a reason,
  choke / surge flags (see :func:`speed_line`).

Surge and choke at meanline level (preliminary-design criteria, labelled as such)
-------------------------------------------------------------------------------
* choke: TurboFlow's impeller-throat choking criterion, or loss of convergence at the
  high-flow end of a speed line (recorded as ``no_convergence``, not silently treated as choke);
* surge: the point of maximum pressure ratio on a speed line (zero slope dPR/dm). Points at
  lower flow have positive slope and are flagged ``surge_side``. This is the classic
  static-stability criterion; real surge can occur earlier or later.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from turboml.physics.compressor import CentrifugalCompressorGeometry

LOSS_SETS = ("oh", "zhang_set1", "zhang_set2", "zhang_set3")

# --------------------------------------------------------------------------------------
# Slip hook (installed once; parameterised by a module-level delta)
# --------------------------------------------------------------------------------------
_SLIP_DELTA = {"value": 0.0}


def wiesner_slip_velocity(u_out: float, trailing_edge_angle_deg: float, z: float) -> float:
    """Wiesner (1967) slip velocity U2*sqrt(cos beta2b)/Z^0.7, beta2b in DEGREES.

    Slip factor sigma = 1 - slip_velocity/U2 (for radial-element exits). The limiting-radius-ratio
    correction of Wiesner is not applied (TurboFlow's formulation doesn't either). It is part of
    the slip uncertainty.
    """
    return u_out * np.sqrt(np.cos(np.radians(trailing_edge_angle_deg))) / z**0.7


def _install_slip_hook() -> None:
    from turboflow.centrifugal_compressor import slip_model as sm

    if getattr(sm.evaluate_slip, "_turboml", False):
        return

    def evaluate_slip(slip_model, inp, geometry):
        if slip_model == "no-slip":
            base = 0.0
        elif slip_model == "wiesner":
            base = wiesner_slip_velocity(
                inp["u_out"], geometry["trailing_edge_angle"], geometry["number_of_blades"]
            )
        else:
            raise ValueError(f"unsupported slip model {slip_model!r}")
        return base - _SLIP_DELTA["value"] * inp["u_out"]  # sigma += delta_sigma

    evaluate_slip._turboml = True
    sm.evaluate_slip = evaluate_slip


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@dataclass
class CompressorPoint:
    speed_rpm: float
    mass_flow_kg_s: float
    converged: bool
    failure_reason: str = ""
    pr_tt: float = np.nan
    eta_tt: float = np.nan  # isentropic total-to-total, fraction (not %)
    power_W: float = np.nan
    choked: bool = False
    slip_factor: float = np.nan
    tip_speed_m_s: float = np.nan
    inlet_rel_mach_tip: float = np.nan
    exit_flow_angle_deg: float = np.nan
    losses: dict = field(default_factory=dict)


class CentrifugalCompressorTF:
    """TurboFlow-based centrifugal compressor (cross-check)."""

    def __init__(
        self,
        geometry: CentrifugalCompressorGeometry,
        loss_set: str = "oh",
        delta_sigma: float = 0.0,
        diffuser_cf: float = 0.02,
        T0_in_K: float = 288.15,
        p0_in_Pa: float = 101325.0,
        fluid: str = "air",
    ):
        if loss_set not in LOSS_SETS:
            raise ValueError(f"loss_set must be one of {LOSS_SETS}")
        _install_slip_hook()
        self.geometry, self.loss_set, self.delta_sigma = geometry, loss_set, delta_sigma
        # Skin-friction coefficient of the vaneless diffuser (TurboFlow's algebraic model: angular
        # momentum decays as exp(-Cf dr / (b cos alpha))). Default 0.02 = TurboFlow example value.
        # Raising it is how a calibration can represent diffuser losses the vaneless model
        # lacks (e.g. the KJ-66 wedge vanes + 90 deg turn). With the "oh" loss set it acts on the
        # diffuser only (the impeller uses its own Jansen friction correlation).
        self.diffuser_cf = diffuser_cf
        self.T0_in_K, self.p0_in_Pa, self.fluid = T0_in_K, p0_in_Pa, fluid
        self._last_x = None  # continuation: last converged solution vector

    # TurboFlow configuration --------------------------------------------------------
    def _simulation_options(self) -> dict:
        lm = {"model": self.loss_set, "loss_coefficient": "static_enthalpy_loss"}
        return {
            "slip_model": "wiesner",
            "loss_model": {"impeller": dict(lm), "vaneless_diffuser": dict(lm)},
            "vaneless_diffuser_model": "algebraic",
            "rel_step_fd": 1e-4,
            "factors": {
                "skin_friction": self.diffuser_cf,
                "wall_heat_flux": 0.0,
                "wake_width": 0.366,
            },
            "choking_criterion": "evaluate_throat",
        }

    @staticmethod
    def _solver_options() -> dict:
        return {"method": "hybr", "tolerance": 1e-8, "max_iterations": 100,
                "derivative_method": "2-point", "derivative_abs_step": 1e-6,
                "print_convergence": False, "plot_convergence": False}  # fmt: skip

    def evaluate(
        self, mass_flow_kg_s: float, speed_rpm: float, initial_guess=None
    ) -> CompressorPoint:
        """One operating point. Never raises for solver failures (failures are data)."""
        from turboflow.centrifugal_compressor import performance_analysis as pa

        op = {"fluid_name": self.fluid, "T0_in": self.T0_in_K, "p0_in": self.p0_in_Pa,
              "mass_flow_rate": mass_flow_kg_s, "omega": speed_rpm * np.pi / 30.0, "alpha_in": 0.0}  # fmt: skip
        guess = initial_guess or self._last_x or {
            "efficiency_impeller": 0.75, "phi_impeller": 0.3, "Ma_vaned_diffuser": 0.5
        }  # fmt: skip
        pt = CompressorPoint(speed_rpm, mass_flow_kg_s, converged=False,
                             tip_speed_m_s=self.geometry.tip_speed_m_s(speed_rpm))  # fmt: skip
        _SLIP_DELTA["value"] = self.delta_sigma
        try:
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
                solver = pa.compute_single_operation_point(
                    op,
                    guess,
                    self.geometry.to_turboflow(),
                    self._simulation_options(),
                    self._solver_options(),
                )
        except Exception as err:  # noqa: BLE001 - recorded as data
            pt.failure_reason = f"no_convergence: {type(err).__name__}: {str(err)[:120]}"
            return pt
        finally:
            _SLIP_DELTA["value"] = 0.0
        if not solver.success:
            pt.failure_reason = "no_convergence"
            return pt
        r = solver.problem.results
        o = r["overall"]
        g = lambda v: float(np.asarray(v).ravel()[0])  # noqa: E731
        imp = r["impeller"]
        pt.converged = True
        pt.pr_tt = g(o["PR_tt"])
        pt.eta_tt = g(o["efficiency_tt"]) / 100.0
        pt.power_W = g(o["power"])
        pt.choked = bool(g(imp["throat_plane"].get("choked", 0.0)))
        ex, inl = imp["exit_plane"], imp["inlet_plane"]
        u2 = g(ex["blade_speed"])
        theta = self.geometry.trailing_edge_angle_deg
        slip_vel = u2 + g(ex["v_m"]) * np.tan(np.radians(theta)) - g(ex["v_t"])
        pt.slip_factor = 1.0 - slip_vel / u2
        pt.exit_flow_angle_deg = g(ex["alpha"])
        w_tip = np.hypot(g(inl["v_m"]), self.geometry.radius_tip_in_m * speed_rpm * np.pi / 30.0)
        pt.inlet_rel_mach_tip = w_tip / g(inl["a"])
        pt.losses = (
            {k: g(v) for k, v in imp.get("loss_dict", {}).items()} if "loss_dict" in imp else {}
        )
        self._last_x = dict(solver.problem.vars_real)
        return pt

    def speed_line(self, speed_rpm: float, mass_flows: np.ndarray) -> pd.DataFrame:
        """Evaluate a speed line by continuation from the middle of ``mass_flows`` outward.

        Columns: CompressorPoint fields + ``surge_side`` (positive dPR/dm, left of peak PR)
        + ``is_peak_pr``. Failures stay in the table with their reason.
        """
        m = np.sort(np.asarray(mass_flows, dtype=float))
        start = len(m) // 2
        rows: dict[int, CompressorPoint] = {}
        for order in (range(start, len(m)), range(start - 1, -1, -1)):
            self._last_x = None if order.start == start else self._x_at_start
            stop = False
            for i in order:
                if stop:
                    rows[i] = CompressorPoint(speed_rpm, m[i], False, "skipped_after_failure",
                                              tip_speed_m_s=self.geometry.tip_speed_m_s(speed_rpm))  # fmt: skip
                    continue
                pt = self.evaluate(m[i], speed_rpm)
                rows[i] = pt
                if i == start:
                    self._x_at_start = self._last_x
                stop = not pt.converged
        df = pd.DataFrame([asdict(rows[i]) | {"losses": None} for i in range(len(m))]).drop(
            columns="losses"
        )
        ok = df.converged
        df["is_peak_pr"] = False
        df["surge_side"] = False
        if ok.any():
            ipk = df.loc[ok, "pr_tt"].idxmax()
            df.loc[ipk, "is_peak_pr"] = True
            df.loc[ok & (df.mass_flow_kg_s < df.loc[ipk, "mass_flow_kg_s"]), "surge_side"] = True
        df["loss_set"], df["delta_sigma"] = self.loss_set, self.delta_sigma
        df["diffuser_cf"] = self.diffuser_cf
        return df

    _x_at_start = None


def surge_margin(m_operating: float, m_surge: float) -> float:
    """Surge margin at constant speed, SM = (m_op - m_surge) / m_op (mass-flow definition)."""
    return (m_operating - m_surge) / m_operating

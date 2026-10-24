"""pyCycle (NASA / OpenMDAO) model of a single-spool micro turbojet, design point.

What this model is
------------------
pyCycle builds the engine from "elements" (flight conditions, inlet, compressor,
combustor, turbine, nozzle, shaft) linked by flow stations, and solves the coupled
equations with a Newton solver. Two balances close the design-point cycle:

* fuel-air ratio FAR is varied until the combustor exit temperature equals the
  requested turbine inlet temperature (TIT);
* turbine pressure ratio is varied until the shaft's net power is zero, i.e.
  η_m · P_turbine = P_compressor + P_offtake (pyCycle's ``fracLoss = 1 − η_m``).

The air mass flow is an input (sizing by mass flow). The model is equivalent to pyCycle's
``simple_turbojet`` example with micro-turbojet changes:

* **convergent** nozzle (``nozzType='CV'``) instead of convergent-divergent;
* mechanical efficiency and optional shaft power off-take;
* **fuel enthalpy set from the LHV and combustion efficiency.** pyCycle's combustor has a
  fuel-enthalpy input (``mix:h``) that defaults to 0, and the examples leave it there.
  With pyCycle's thermo data, h_fuel = 0 means an effective LHV of ≈44.8 MJ/kg for
  C12H23, about 4–5 % above real Jet A (≥42.8 MJ/kg). That would bias TSFC low by about the
  same amount. We set it explicitly via :func:`turboml.gas.thermo.fuel_enthalpy_J_per_kg`
  (see docs/DECISIONS.md D-010);
* SI inputs/outputs (pyCycle works in US units internally; OpenMDAO converts).

Compressor and turbine maps
---------------------------
At the design point a map only fixes its own scaling; the cycle result does not depend
on map shape. pyCycle's generic AXI5 (axial compressor) and LPT2269 maps are therefore
placeholders here. Off-design needs real centrifugal-compressor and turbine maps, which
come from the meanline models in Phase 2/3. **Do not use this model off-design yet.**

Thermodynamics
--------------
NASA CEA chemical-equilibrium thermo only. pyCycle's TABULAR option was dropped (D-016):
in the Phase 1 sweep its table interpolation produced up to 1 % error in compressor work
and 3.5 % in TSFC.

Role in TurboML
---------------
This model is the *reference* for cross-checks (:mod:`turboml.physics.crosscheck`). Training
data come from the fast 0-D model (:mod:`turboml.physics.cycle_simple`).
"""

from __future__ import annotations

import warnings

import numpy as np
import openmdao.api as om
import pycycle.api as pyc
from openmdao.core.analysis_error import AnalysisError

from turboml.gas.thermo import fuel_enthalpy_J_per_kg
from turboml.physics.atmosphere import geopotential_m
from turboml.physics.cycle_common import CycleInputs, CycleResult, Station
from turboml.physics.cycle_simple import design_point as simple_design_point

MACH_MIN = 1e-6  # pyCycle's flight-condition element is singular at exactly Mach 0
DESIGN_MN = {"inlet": 0.4, "comp": 0.15, "burner": 0.05, "turb": 0.3}  # static calcs only


class MicroTurbojet(pyc.Cycle):
    """Design-point single-spool turbojet with a convergent nozzle."""

    def setup(self):
        self.options["thermo_method"] = "CEA"
        self.options["thermo_data"] = pyc.species_data.janaf
        fuel = "Jet-A(g)"

        self.add_subsystem("fc", pyc.FlightConditions())
        self.add_subsystem("inlet", pyc.Inlet())
        self.add_subsystem(
            "comp", pyc.Compressor(map_data=pyc.AXI5, map_extrap=True), promotes_inputs=["Nmech"]
        )
        self.add_subsystem("burner", pyc.Combustor(fuel_type=fuel))
        self.add_subsystem("turb", pyc.Turbine(map_data=pyc.LPT2269), promotes_inputs=["Nmech"])
        self.add_subsystem("nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))
        self.add_subsystem("shaft", pyc.Shaft(num_ports=2), promotes_inputs=["Nmech"])
        self.add_subsystem("perf", pyc.Performance(num_nozzles=1, num_burners=1))

        self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I", connect_w=False)
        self.pyc_connect_flow("inlet.Fl_O", "comp.Fl_I")
        self.pyc_connect_flow("comp.Fl_O", "burner.Fl_I")
        self.pyc_connect_flow("burner.Fl_O", "turb.Fl_I")
        self.pyc_connect_flow("turb.Fl_O", "nozz.Fl_I")

        # Air mass flow is a free input (not a balance), so declare its units here.
        self.set_input_defaults("inlet.Fl_I:stat:W", 1.0, units="kg/s")

        self.connect("comp.trq", "shaft.trq_0")
        self.connect("turb.trq", "shaft.trq_1")
        self.connect("fc.Fl_O:stat:P", "nozz.Ps_exhaust")
        self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
        self.connect("comp.Fl_O:tot:P", "perf.Pt3")
        self.connect("burner.Wfuel", "perf.Wfuel_0")
        self.connect("inlet.F_ram", "perf.ram_drag")
        self.connect("nozz.Fg", "perf.Fg_0")

        bal = self.add_subsystem("balance", om.BalanceComp())
        bal.add_balance(
            "FAR", eq_units="degK", lower=1e-4, upper=0.06, val=0.02, rhs_name="T4_target"
        )
        self.connect("balance.FAR", "burner.Fl_I:FAR")
        self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")
        bal.add_balance("turb_PR", val=2.0, lower=1.001, upper=8.0, eq_units="W", rhs_val=0.0)
        self.connect("balance.turb_PR", "turb.PR")
        self.connect("shaft.pwr_net", "balance.lhs:turb_PR")

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options.update(
            {
                "atol": 1e-8,
                "rtol": 1e-10,
                "iprint": -1,
                "maxiter": 30,
                "solve_subsystems": True,
                "max_sub_solves": 100,
                "err_on_non_converge": True,
                "reraise_child_analysiserror": False,
            }
        )
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement="scalar", iprint=-1)
        self.linear_solver = om.DirectSolver()
        super().setup()


class PyCycleDesignModel:
    """Builds the OpenMDAO problem once; :meth:`evaluate` reruns it for new inputs."""

    def __init__(self):
        self.prob = om.Problem(reports=False)
        self.prob.model = MicroTurbojet()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)
        # Locate the combustor's fuel-enthalpy input (not promoted by pyCycle).
        (self._fuel_h_name,) = [
            n
            for n in self.prob.model.get_io_metadata(iotypes="input")
            if n.startswith("burner.mix_fuel.") and n.endswith("mix:h")
        ]

    def evaluate(self, x: CycleInputs) -> CycleResult:
        p = self.prob
        guess = simple_design_point(x)  # independent 0-D model -> Newton initial guess

        # pyCycle's US-1976 table is indexed by geopotential altitude; our inputs are geometric.
        p.set_val("fc.alt", geopotential_m(x.alt_m), units="m")
        p.set_val("fc.MN", max(x.mach, MACH_MIN))
        p.set_val("fc.dTs", x.dT_isa_K, units="degK")
        p.set_val("inlet.Fl_I:stat:W", x.W_air_kg_s, units="kg/s")
        p.set_val("inlet.ram_recovery", x.inlet_recovery)
        p.set_val("comp.PR", x.pr_c)
        p.set_val("comp.eff", x.eta_c)
        p.set_val("burner.dPqP", x.dp_burner_frac)
        p.set_val(
            self._fuel_h_name,
            fuel_enthalpy_J_per_kg(x.lhv_J_kg, x.eta_burner),
            units="J/kg",
        )
        p.set_val("balance.T4_target", x.tit_K, units="degK")
        p.set_val("turb.eff", x.eta_t)
        p.set_val("nozz.Cv", x.nozzle_cv)
        p.set_val("nozz.dPqP", x.dp_nozzle_frac)
        p.set_val("shaft.fracLoss", 1.0 - x.eta_mech)
        p.set_val("shaft.HPX", x.power_offtake_W, units="W")
        p.set_val("Nmech", 1.0e5, units="rpm")  # irrelevant at design point (map scaling only)
        for el, mn in DESIGN_MN.items():
            p.set_val(f"{el}.MN", mn)
        if guess.converged:
            p.set_val("balance.FAR", guess.far)
            p.set_val("balance.turb_PR", guess.pr_t)

        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p.run_model()
        except AnalysisError as err:
            return CycleResult(
                model="pycycle_cea",
                converged=False,
                failure_reason=f"no_convergence: {str(err).splitlines()[0][:200]}",
            )
        return self._result(x)

    def _get(self, name: str, units: str | None = None) -> float:
        return float(np.asarray(self.prob.get_val(name, units=units)).ravel()[0])

    def _result(self, x: CycleInputs) -> CycleResult:
        g = self._get
        st = {
            k: Station(g(f"{el}.Fl_O:tot:T", "degK"), g(f"{el}.Fl_O:tot:P", "Pa"))
            for k, el in [
                ("0", "fc"),
                ("2", "inlet"),
                ("3", "comp"),
                ("4", "burner"),
                ("5", "turb"),
                ("9", "nozz"),
            ]
        }
        Fn, Wf = g("perf.Fn", "N"), g("burner.Wfuel", "kg/s")
        return CycleResult(
            model="pycycle_cea",
            converged=True,
            W_air_kg_s=g("inlet.Fl_O:stat:W", "kg/s"),
            Fn_N=Fn,
            Fg_N=g("nozz.Fg", "N"),
            F_ram_N=g("inlet.F_ram", "N"),
            Wf_kg_s=Wf,
            TSFC_kg_N_s=Wf / Fn if Fn > 0 else float("nan"),
            far=g("balance.FAR"),
            pr_t=g("balance.turb_PR"),
            P_comp_W=-g("comp.power", "W"),
            P_turb_W=g("turb.power", "W"),
            V9_m_s=g("nozz.Fl_O:stat:V", "m/s"),
            Ps9_Pa=g("nozz.Fl_O:stat:P", "Pa"),
            A9_m2=g("nozz.Fl_O:stat:area", "m**2"),
            # pyCycle's nozz.mux.choked output is never assigned in Mux.compute(), so it is
            # meaningless; a convergent nozzle is choked iff its exit static > ambient
            # (tolerance well above the Newton tolerance, so threshold cases don't flicker).
            choked=bool(g("nozz.Fl_O:stat:P", "Pa") > g("fc.Fl_O:stat:P", "Pa") * (1 + 1e-4)),
            stations=st,
        )

    def ambient(self) -> tuple[float, float]:
        """Static (T [K], p [Pa]) of the free stream of the last evaluation."""
        return self._get("fc.Fl_O:stat:T", "degK"), self._get("fc.Fl_O:stat:P", "Pa")

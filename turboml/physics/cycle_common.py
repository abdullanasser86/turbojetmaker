"""Shared input / result types for the design-point cycle models.

Both cycle models (pyCycle wrapper and the independent 0-D model) take the same
:class:`CycleInputs` and return the same :class:`CycleResult`, so they can be compared
field by field. Component parameters have no defaults on purpose: their values come
from YAML (``configs/cycle_baseline.yaml``) with stated provenance.

Station numbering (SAE ARP755 style, simplified for a single-spool turbojet):

    0  free stream          2  compressor inlet (after intake)
    3  compressor exit      4  turbine inlet (combustor exit)
    5  turbine exit         9  nozzle exit (= throat for a convergent nozzle)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from turboml.config import CONFIG_DIR


@dataclass
class CycleInputs:
    """Design-point inputs. SI units. Efficiencies are isentropic total-to-total."""

    W_air_kg_s: float  # air mass flow at the intake
    pr_c: float  # compressor total pressure ratio Pt3/Pt2
    eta_c: float  # compressor isentropic efficiency
    tit_K: float  # turbine inlet total temperature Tt4
    eta_t: float  # turbine isentropic efficiency
    eta_mech: float  # shaft: P_comp + P_offtake = eta_mech * P_turb
    eta_burner: float  # combustion efficiency (fraction of LHV released)
    dp_burner_frac: float  # combustor total pressure loss (Pt3 - Pt4)/Pt3
    inlet_recovery: float  # intake total pressure recovery Pt2/Pt0
    dp_nozzle_frac: float  # jet pipe + nozzle total pressure loss (Pt5 - Pt9)/Pt5
    nozzle_cv: float  # nozzle velocity coefficient (multiplies momentum thrust)
    lhv_J_kg: float  # fuel lower heating value
    power_offtake_W: float = 0.0  # shaft power extraction (generator, pumps)
    # --- flight condition
    alt_m: float = 0.0
    mach: float = 0.0
    dT_isa_K: float = 0.0
    # Override ambient statics (used to feed both models identical ambient conditions).
    Ts0_K: float | None = None
    Ps0_Pa: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path = CONFIG_DIR / "cycle_baseline.yaml", **overrides):
        """Build inputs from the ``values`` block of a cycle YAML file, plus overrides."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        vals = {k: v["value"] if isinstance(v, dict) else v for k, v in raw["values"].items()}
        vals.update(overrides)
        bad = {k: v for k, v in vals.items() if isinstance(v, str)}
        if bad:  # e.g. PyYAML reads '42.8e6' (no exponent sign) as a string
            raise TypeError(f"non-numeric cycle inputs in {path}: {bad}")
        names = {f.name for f in fields(cls)}
        unknown = set(vals) - names
        if unknown:
            raise KeyError(f"unknown cycle inputs: {sorted(unknown)}")
        return cls(**vals)


@dataclass
class Station:
    Tt_K: float
    Pt_Pa: float


@dataclass
class CycleResult:
    model: str
    converged: bool
    failure_reason: str = ""
    W_air_kg_s: float = float("nan")
    Fn_N: float = float("nan")  # net thrust
    Fg_N: float = float("nan")  # gross thrust
    F_ram_N: float = float("nan")  # ram drag
    Wf_kg_s: float = float("nan")  # fuel flow
    TSFC_kg_N_s: float = float("nan")
    far: float = float("nan")  # fuel-air ratio Wf/W_air
    pr_t: float = float("nan")  # turbine total pressure ratio Pt4/Pt5
    P_comp_W: float = float("nan")
    P_turb_W: float = float("nan")
    V9_m_s: float = float("nan")  # nozzle exit velocity
    Ps9_Pa: float = float("nan")
    A9_m2: float = float("nan")  # nozzle exit (throat) area
    choked: bool = False
    stations: dict[str, Station] = field(default_factory=dict)

    @property
    def EGT_K(self) -> float:
        """Exhaust gas temperature, taken as turbine exit total temperature Tt5."""
        return self.stations["5"].Tt_K

    @property
    def specific_thrust_N_s_kg(self) -> float:
        return self.Fn_N / self.W_air_kg_s

    @property
    def TSFC_g_kN_s(self) -> float:
        return self.TSFC_kg_N_s * 1e6

    def to_flat_dict(self) -> dict:
        d = asdict(self)
        for k, s in d.pop("stations").items():
            d[f"Tt{k}_K"], d[f"Pt{k}_Pa"] = s["Tt_K"], s["Pt_Pa"]
        return d

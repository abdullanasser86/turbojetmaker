"""Second, independent centrifugal-compressor meanline model: RadComp (cross-check only).

RadComp (Picard, Schiffmann & Ahmed 2023; github.com/cyrilpic/radcomp, v1.0.1) implements the
meanline model of Schiffmann & Favrat (Energy 35(1), 2010) with its own loss set and a
Wiesner-Busemann slip factor, and a vaneless diffuser. It shares no loss correlations with
TurboFlow's Oh/Zhang sets, so agreement between the two is evidence the trends are physical.
Disagreement is evidence of loss-model (epistemic) uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from turboml.config import CONFIG_DIR
from turboml.physics.compressor import CentrifugalCompressorGeometry


def radcomp_geometry(
    g: CentrifugalCompressorGeometry,
    path: str | Path = CONFIG_DIR / "components" / "kj66_compressor.yaml",
):
    from radcompressor.geometry import Geometry

    ex = {k: v["value"] for k, v in yaml.safe_load(Path(path).read_text())["radcomp_extra"].items()}
    return Geometry(
        r1=g.radius_tip_in_m, r2s=g.radius_tip_in_m, r2h=g.radius_hub_in_m,
        beta2=ex["mid_inlet_angle_deg"], beta2s=ex["shroud_inlet_angle_deg"], alpha2=0.0,
        r4=g.radius_out_m, b4=g.width_out_m, r5=g.diffuser_radius_out_m, b5=g.diffuser_width_out_m,
        beta4=g.trailing_edge_angle_deg, n_blades=g.number_of_blades, n_splits=0,
        blade_e=ex["blade_thickness_m"], rug_imp=ex["roughness_m"], clearance=g.tip_clearance_m,
        backface=ex["backface_clearance_m"], rug_ind=ex["roughness_m"], l_ind=ex["inducer_length_m"],
        l_comp=g.length_axial_m, blockage=[1.0, 1.0, 1.0, 1.0, 1.0],
    )  # fmt: skip


def evaluate(g: CentrifugalCompressorGeometry, mass_flow_kg_s: float, speed_rpm: float,
             T0_K: float = 288.15, p0_Pa: float = 101325.0) -> dict:  # fmt: skip
    """One RadComp operating point -> dict(converged, pr_tt, eta_tt, failure_reason)."""
    from radcompressor.compressor import Compressor
    from radcompressor.condition import OperatingCondition
    from radcompressor.thermo import CoolPropFluid

    in0 = CoolPropFluid("Air").thermo_prop("PT", p0_Pa, T0_K)
    op = OperatingCondition(in0=in0, fld=in0.fld, m=mass_flow_kg_s, n_rot=speed_rpm * np.pi / 30)
    comp = Compressor(radcomp_geometry(g), op)
    try:
        with np.errstate(all="ignore"):
            ok = bool(comp.calculate())
    except Exception as err:  # noqa: BLE001 - recorded as data
        return {
            "converged": False,
            "failure_reason": f"no_convergence: {err!r}"[:120],
            "pr_tt": np.nan,
            "eta_tt": np.nan,
        }
    if not ok:
        return {
            "converged": False,
            "failure_reason": "radcomp_invalid",
            "pr_tt": np.nan,
            "eta_tt": np.nan,
        }
    return {
        "converged": True,
        "failure_reason": "",
        "pr_tt": float(comp.PR),
        "eta_tt": float(comp.eff),
    }

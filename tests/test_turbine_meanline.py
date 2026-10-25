"""Fast axial-turbine meanline model: port verification against TurboFlow, robustness, failure labels."""

import math

import pandas as pd
import pytest

from turboml.gas.thermo import products
from turboml.physics.turbine import KOFSKEY_1972_CONFIG
from turboml.physics.turbine_meanline import (
    AxialStageGeometry,
    AxialTurbineMeanline,
    benner_losses,
    critical_mach,
)

GEOM = AxialStageGeometry.from_turboflow_yaml(KOFSKEY_1972_CONFIG)
TF_MAP = pd.read_csv("validation/phase2/kofskey_map.csv")


@pytest.fixture(scope="module")
def turb():
    return AxialTurbineMeanline(GEOM, incidence_sign="turboflow")


@pytest.mark.parametrize(
    "speed_pct,pr", [(100, 2.25), (100, 3.5), (50, 1.8), (50, 3.0), (110, 2.0), (70, 4.0)]
)
def test_matches_turboflow_kofskey_map(turb, speed_pct, pr):
    """Same equations as TurboFlow 0.1.18 -> same answers (subsonic and choked points)."""
    ref = TF_MAP[(TF_MAP.speed_pct_design == speed_pct) & (TF_MAP.pr_ts.round(3) == pr)].iloc[0]
    p = turb.evaluate(295.6, 1.38e5, 1627.0 * speed_pct / 100, pr)
    assert p.converged
    assert p.mass_flow_kg_s == pytest.approx(ref.mass_flow_kg_s, rel=2e-3)
    assert p.eta_ts == pytest.approx(ref.eta_ts, abs=1e-3)
    assert p.power_W == pytest.approx(ref.power_W, rel=2e-3)
    assert abs(p.p_error_rel) < 1e-7


def test_benner_port_equals_turboflow_function():
    """Loss-coefficient port: identical to TurboFlow's loss_model_benner for the same inputs."""
    from turboflow.axial_turbine import loss_model_benner as br

    c = GEOM.cascades()[1]  # rotor
    f = {"Re_in": 3.1e5, "Re_out": 4.2e5, "Ma_rel_in": 0.31, "Ma_rel_out": 0.74, "p0_rel_is": 1.05e5, "p_in": 9.6e4,
         "p0_rel_out": 1.0e5, "p_out": 7.3e4, "beta_in": 36.0, "beta_out": -61.0, "gamma_out": 1.401,
         "w_out": 250.0, "h_out": 0.0, "h_is": 0.0, "p0_rel_in": 1.06e5}  # fmt: skip
    geom = {"hub_tip_ratio_in": c.hub_tip_in, "pitch": c.pitch, "chord": c.chord, "leading_edge_angle": c.theta_in,
            "maximum_thickness": c.t_max, "cascade_type": "rotor", "trailing_edge_thickness": c.t_te,
            "opening": c.opening, "height": c.height, "stagger_angle": c.stagger, "tip_clearance": c.tip_clearance,
            "leading_edge_diameter": c.le_diameter, "leading_edge_wedge_angle": c.wedge, "A_throat": c.A_throat,
            "A_out": c.A_out, "axial_chord": c.axial_chord}  # fmt: skip
    ref = br.compute_losses(
        {
            "flow": f,
            "geometry": geom,
            "loss_model": {"inlet_displacement_thickness_height_ratio": 0.011},
        }
    )
    ours = benner_losses(c, f, incidence_sign="turboflow")
    for k in ("profile", "trailing", "secondary", "clearance", "incidence", "total"):
        assert ours[k] == pytest.approx(float(ref[f"loss_{k}"]), rel=1e-9, abs=1e-12), k


def test_critical_mach_limits():
    assert critical_mach(1.4, 1.0) == pytest.approx(1.0, abs=1e-9)
    assert 0.9 < critical_mach(1.4, 0.9) < 1.0


def test_warm_start_does_not_change_the_answer():
    cold = AxialTurbineMeanline(GEOM, warm_start=False).evaluate(295.6, 1.38e5, 1627.0, 2.5)
    warm = AxialTurbineMeanline(GEOM)
    for pr in (1.8, 3.2, 2.2):
        warm.evaluate(295.6, 1.38e5, 1627.0, pr)
    p = warm.evaluate(295.6, 1.38e5, 1627.0, 2.5)
    assert p.mass_flow_kg_s == pytest.approx(cold.mass_flow_kg_s, rel=1e-7)
    assert p.eta_ts == pytest.approx(cold.eta_ts, abs=1e-7)


def test_combustion_products_and_hot_inlet():
    t = AxialTurbineMeanline(GEOM, gas=products(0.02))
    p = t.evaluate(1150.0, 2.5e5, 1627.0 * math.sqrt(1150 / 295.6), 2.0)
    assert p.converged and 0.6 < p.eta_ts < 0.95


def test_failures_are_labelled(turb):
    assert turb.evaluate(295.6, 1.38e5, 1627.0, 0.9).failure_reason == "invalid_input"
    p = turb.evaluate(295.6, 1.38e5, 1627.0, 30.0)  # far beyond limit loading
    assert not p.converged and p.failure_reason in ("limit_loading", "no_convergence")


def test_rotor_plus_option_changes_only_rotor_incidence(turb):
    alt = AxialTurbineMeanline(GEOM, incidence_sign="rotor_plus")
    a, b = turb.evaluate(295.6, 1.38e5, 1627.0, 2.3), alt.evaluate(295.6, 1.38e5, 1627.0, 2.3)
    assert a.losses["stator"]["incidence"] == pytest.approx(
        b.losses["stator"]["incidence"], abs=1e-6
    )
    assert a.losses["rotor"]["incidence"] != pytest.approx(b.losses["rotor"]["incidence"], abs=1e-4)
    with pytest.raises(ValueError):
        AxialTurbineMeanline(GEOM, incidence_sign="nonsense")

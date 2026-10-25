"""Axial turbine sizing (design mode)."""

import math

import pytest

from turboml.gas.thermo import products
from turboml.physics.turbine_design import design_stage, size_axial_stage

OMEGA = 130000 * math.pi / 30
ARGS = (0.306, 1150.0, 2.68e5, OMEGA, 140e3, 0.7, 1.0, 0.5)


def test_velocity_triangles():
    s = size_axial_stage(*ARGS, gas=products(0.02))
    assert s.U_m_s == pytest.approx(math.sqrt(140e3 / 1.0))
    assert s.r_mean_m == pytest.approx(s.U_m_s / OMEGA)
    a = s.angles_deg
    # psi = 1, R = 0.5: C_theta2 = U, C_theta3 = 0 -> zero exit swirl, symmetric triangles
    assert a["alpha3"] == pytest.approx(0.0, abs=1e-9)
    assert a["alpha2"] == pytest.approx(-a["beta3"], abs=1e-9)
    assert not s.warnings


def test_design_stage_hits_flow_and_work():
    s, t, p = design_stage(*ARGS, gas=products(0.02))
    assert p.converged
    assert p.mass_flow_kg_s == pytest.approx(0.306, rel=1e-5)
    assert p.work_J_kg == pytest.approx(140e3, rel=1e-5)


def test_infeasible_sizing_raises():
    with pytest.raises(ValueError):
        size_axial_stage(
            5.0, 1150.0, 2.68e5, OMEGA, 140e3, 0.2, 1.0, 0.5
        )  # blades taller than the radius

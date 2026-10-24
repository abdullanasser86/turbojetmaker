"""pyCycle wrapper: agreement with the 0-D model and correct use of fuel properties."""

from dataclasses import replace

import pytest

from turboml.physics.cycle_common import CycleInputs
from turboml.physics.cycle_simple import design_point

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cea():
    from turboml.physics.cycle_pycycle import PyCycleDesignModel

    return PyCycleDesignModel()


def test_baseline_matches_simple_model(cea):
    x = CycleInputs.from_yaml()
    rp = cea.evaluate(x)
    Ts, Ps = cea.ambient()
    rs = design_point(replace(x, Ts0_K=Ts, Ps0_Pa=Ps))
    assert rp.converged and rs.converged
    assert rp.Fn_N == pytest.approx(rs.Fn_N, rel=2e-3)
    assert rp.TSFC_kg_N_s == pytest.approx(rs.TSFC_kg_N_s, rel=2e-3)
    assert rp.P_comp_W == pytest.approx(rs.P_comp_W, rel=1e-5)
    assert rp.choked == rs.choked


def test_fuel_enthalpy_is_applied(cea):
    """If the LHV/eta_b input were ignored (pyCycle default h_fuel = 0), fuel flow would not
    respond. Fuel flow must scale ~ 1/(eta_b * LHV) at fixed TIT."""
    x = CycleInputs.from_yaml(eta_burner=1.0)
    wf1 = cea.evaluate(x).Wf_kg_s
    wf2 = cea.evaluate(replace(x, eta_burner=0.9)).Wf_kg_s
    assert wf2 / wf1 == pytest.approx(1 / 0.9, rel=0.01)


def test_nonconvergence_is_reported_not_raised(cea):
    r = cea.evaluate(CycleInputs.from_yaml(eta_c=0.35, eta_t=0.35))
    assert not r.converged and r.failure_reason.startswith("no_convergence")
    # model stays usable after a failure
    assert cea.evaluate(CycleInputs.from_yaml()).converged

import pytest

from turboml.physics.atmosphere import isa


def test_isa_reference_points():
    sl = isa(0.0)
    assert (sl.T_K, sl.p_Pa) == (288.15, 101325.0)
    # Tropopause (11 km geopotential): 216.65 K, 22632 Pa (US Std Atmosphere 1976)
    h11_geometric = 11000.0 * 6356766.0 / (6356766.0 - 11000.0)
    tp = isa(h11_geometric)
    assert tp.T_K == pytest.approx(216.65, abs=1e-6)
    assert tp.p_Pa == pytest.approx(22632.1, abs=1.0)
    assert isa(3000.0, dT_isa_K=15.0).T_K == pytest.approx(isa(3000.0).T_K + 15.0)
    with pytest.raises(ValueError):
        isa(25000.0)


@pytest.mark.slow
@pytest.mark.parametrize("alt_m", [0.0, 3000.0, 6000.0])
def test_isa_matches_pycycle_table(alt_m):
    """Our formula vs pyCycle's tabulated US-1976 atmosphere (Akima interpolation of a
    1000-ft table). The wrapper converts geometric -> geopotential altitude; without that
    conversion the pressure at 6 km would be off by 0.08 %."""
    from turboml.physics.cycle_common import CycleInputs
    from turboml.physics.cycle_pycycle import PyCycleDesignModel

    m = PyCycleDesignModel("TABULAR")
    m.evaluate(CycleInputs.from_yaml(alt_m=alt_m))
    T, p = m.ambient()
    ref = isa(alt_m)
    assert T == pytest.approx(ref.T_K, rel=1e-4)
    assert p == pytest.approx(ref.p_Pa, rel=1e-4)

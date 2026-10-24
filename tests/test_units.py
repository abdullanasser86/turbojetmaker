import math

import pytest

from turboml import units as u


def test_round_trips():
    for f, g, x in [
        (u.rpm_to_rad_s, u.rad_s_to_rpm, 117_000.0),
        (u.degR_to_K, u.K_to_degR, 518.67),
        (u.psi_to_Pa, u.Pa_to_psi, 14.696),
        (u.lbm_s_to_kg_s, u.kg_s_to_lbm_s, 2.2),
        (u.lbf_to_N, u.N_to_lbf, 11800.0),
    ]:
        assert g(f(x)) == pytest.approx(x, rel=1e-14)


def test_known_values():
    assert u.rpm_to_rad_s(60.0) == pytest.approx(2 * math.pi)
    assert u.degR_to_K(518.67) == pytest.approx(288.15, abs=1e-9)  # ISA SLS
    assert u.psi_to_Pa(14.6959488) == pytest.approx(101325.0, rel=1e-8)
    assert u.lbf_to_N(1.0) == pytest.approx(u.LBM_TO_KG * u.G0, rel=1e-12)


def test_tsfc_conversion():
    # 1 lbm/(lbf h) = 0.45359237 / (4.4482216 * 3600) kg/(N s) = 28.3253 g/(kN s)
    tsfc = u.tsfc_lbm_per_lbf_h_to_kg_per_N_s(1.0)
    assert u.tsfc_kg_per_N_s_to_g_per_kN_s(tsfc) == pytest.approx(28.3253, rel=1e-5)

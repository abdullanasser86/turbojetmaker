"""Mechanical proxies: creep-rupture data and checks, blade stresses, envelope, calibration band."""

import math

import pandas as pd
import pytest
import yaml

from turboml.config import load_config
from turboml.physics import creep
from turboml.physics import mechanical as mech

MATS = mech.MaterialSet.from_yaml()


def test_rupture_table_transcription():
    """NiDI No. 393 Tables 9/10: the ksi and printed MPa columns agree (except the documented typo)."""
    d = pd.read_csv(creep.RUPTURE_CSV, comment="#")
    rel = (d.strength_MPa_printed / (d.strength_ksi * 6.894757) - 1).abs()
    assert (rel < 0.025).all()
    typo = d[(d.alloy == "in713c") & (d.life_h == 1000) & (d.T_F == 1400)].iloc[0]
    assert typo.strength_MPa_printed == 458 and typo.strength_ksi == 65


def test_strength_reproduces_table_and_is_monotonic():
    rc = creep.rupture_curve("in713c")
    s, how = rc.strength_Pa((1600 - 32) * 5 / 9 + 273.15, 100.0)  # 1600 F, 100 h: 42 ksi
    assert how == "table" and s == pytest.approx(42 * 6.894757e6, rel=1e-9)
    temps = [1050, 1100, 1150, 1200, 1250, 1300, 1350]
    vals = [rc.strength_Pa(T, 100.0)[0] for T in temps]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
    s1000 = rc.strength_Pa(1150.0, 1000.0)[0]
    s300 = rc.strength_Pa(1150.0, 300.0)[0]
    assert rc.strength_Pa(1150.0, 100.0)[0] > s300 > s1000  # longer life -> lower strength


def test_no_extrapolation_above_data():
    s, how = creep.rupture_curve("marm247").strength_Pa(1400.0, 100.0)
    assert s is None and how == "above_material_data"


def test_allowable_temperature_inverts_strength():
    rc = creep.rupture_curve("in713c")
    T, how = rc.allowable_temperature_K(250e6, 100.0)
    assert how == "ok"
    assert rc.strength_Pa(T, 100.0)[0] == pytest.approx(250e6, rel=1e-6)


def test_root_stress_formula():
    rho, om, rt, rh = 7910.0, 13600.0, 0.035, 0.021
    A = math.pi * (rt**2 - rh**2)
    assert mech.blade_root_stress_Pa(rho, om, A) == pytest.approx(rho * om**2 * (rt**2 - rh**2) / 2)


def test_section_modulus_parabolic_profile():
    # Zs = (8/105) c t^2 derived for y_t = 2 t xi (1 - xi); check by numerical integration of I = int (2y)^3/12 dx
    c, t = 0.01, 0.0015
    n = 20000
    inertia = sum(
        (2 * 2 * t * (k + 0.5) / n * (1 - (k + 0.5) / n)) ** 3 / 12 * c / n for k in range(n)
    )
    assert creep.section_modulus_m3(c, t) == pytest.approx(inertia / (t / 2), rel=1e-6)


def test_creep_feasibility_and_labels():
    # low stress, moderate temperatures: default material passes
    r = mech.evaluate(1150.0, 450.0, 10000.0, 0.03, 5e-4, 500.0, 1050.0, 10e6, 10e6, MATS)
    assert r.feasible and r.material == "in713c" and not r.premium_material_required
    # high centrifugal stress: AN2 is the active constraint even for the premium alloy
    r = mech.evaluate(1150.0, 450.0, 14000.0, 0.03, 4e-3, 500.0, 1088.0, 10e6, 10e6, MATS)
    assert not r.feasible and "AN2" in r.infeasible_reasons
    # NGV hotter than any tabulated data even at PF = 0
    r = mech.evaluate(1420.0, 450.0, 10000.0, 0.03, 5e-4, 500.0, 1050.0, 10e6, 10e6, MATS)
    assert not r.feasible and any(c.startswith("TIT_ngv") for c in r.infeasible_reasons)
    # pattern factor unsourced -> NGV reported as unchecked with PF_max, not silently passed
    r = mech.evaluate(1150.0, 450.0, 10000.0, 0.03, 5e-4, 500.0, 1050.0, 10e6, 10e6, MATS)
    assert any("pattern factor not sourced" in t for t in r.tags)
    assert mech.TURBINE_EFFICIENCY_FLAG in r.tags


def test_premium_material_selected_when_default_fails():
    m = mech.MaterialSet.from_yaml()
    # rotor stress that 713C can't carry at this temperature but MAR-M 247 can
    T = 1100.0
    s713 = creep.rupture_curve("in713c").strength_Pa(T, 100.0)[0]
    s247 = creep.rupture_curve("marm247").strength_Pa(T, 100.0)[0]
    sigma_gb = 0.5 * (s713 + s247) / 1.5  # pure gas-bending load between the two allowables
    r = mech.evaluate(1150.0, 450.0, 1.0, 0.03, 1e-9, 500.0, T, sigma_gb, 1e6, m)
    assert r.feasible and r.material == "marm247" and r.premium_material_required


def test_tip_speed_and_envelope():
    lim = load_config().constraints.max_impeller_tip_speed_m_s
    r = mech.evaluate(
        1150.0, 450.0, 20000.0, 0.03, 5e-4, lim, 1050.0, 10e6, 10e6, MATS
    )  # U2 = 600 m/s
    assert "tip_speed" in r.infeasible_reasons
    env = r.checks["envelope"]
    assert env["diameter_m"] == pytest.approx(1.627 * 0.06)
    assert env["length_m"] == pytest.approx(2.59 * env["diameter_m"])


def test_tit_bound_raised_to_1350():
    assert load_config().bounds["tit_K"].upper == 1350.0


def test_compressor_calibration_split_and_bands():
    cal = yaml.safe_load(open("configs/components/kj66_compressor_calibration.yaml"))
    fit = cal["fit"]
    assert set(fit["calibration_points"]["speed_rpm"]).isdisjoint(
        fit["validation_points"]["speed_rpm"]
    )
    for k in ("delta_sigma", "diffuser_cf"):
        lo, hi = cal[k]["band_2sigma"]
        ilo, ihi = cal[k]["band_inflated"]
        assert ilo < lo < cal[k]["value"] < hi < ihi
    assert cal["phase4_sampling"]["fraction_of_designs"] == 0.2

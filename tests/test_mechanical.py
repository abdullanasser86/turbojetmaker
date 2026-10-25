"""Mechanical proxies: material temperature limit, blade root stress, envelope."""

import math

import pytest
import yaml

from turboml.config import load_config
from turboml.physics import mechanical as mech

MATS = mech.MaterialSet.from_yaml()


def test_materials_are_sourced():
    raw = yaml.safe_load(open("configs/materials.yaml"))
    for m in raw["materials"].values():
        for k in ("density_kg_m3", "max_use_temperature_K", "allowable_stress_Pa"):
            assert m[k][
                "source"
            ], k  # every entry, including the missing ones, says where it comes from


@pytest.mark.parametrize("tit,material,premium,feasible", [
    (1200.0, "in713c", False, True),
    (1265.0, "marm247", True, True),
    (1340.0, None, False, False),
])  # fmt: skip
def test_temperature_check(tit, material, premium, feasible):
    t = mech.turbine_temperature_check(tit, MATS)
    assert (t["material"], t["premium_material_required"], t["feasible"]) == (
        material,
        premium,
        feasible,
    )


def test_design_above_all_limits_is_labelled_not_dropped():
    r = mech.evaluate(1340.0, 13600.0, 0.03, 1e-3, 500.0, MATS)
    assert not r.feasible and "constraint_material_temperature" in r.infeasible_reasons


def test_root_stress_formula():
    # sigma = rho omega^2 (rt^2 - rh^2) / 2 for a constant-section blade
    rho, om, rt, rh = 7913.0, 13600.0, 0.035, 0.021
    A = math.pi * (rt**2 - rh**2)
    assert mech.blade_root_stress_Pa(rho, om, A) == pytest.approx(rho * om**2 * (rt**2 - rh**2) / 2)


def test_missing_allowable_is_unchecked_not_guessed():
    r = mech.evaluate(1150.0, 13600.0, 0.03, 1e-3, 500.0, MATS)
    assert r.checks["blade_root_stress"]["status"] == "unchecked: material_data_missing"
    assert r.feasible  # unchecked is not infeasible


def test_tip_speed_and_envelope():
    lim = load_config().constraints.max_impeller_tip_speed_m_s
    r = mech.evaluate(1150.0, 20000.0, 0.03, 1e-3, lim, MATS)  # U2 = 600 m/s
    assert "constraint_impeller_tip_speed" in r.infeasible_reasons
    env = r.checks["envelope"]
    assert env["diameter_m"] == pytest.approx(1.627 * 0.06)
    assert env["length_m"] == pytest.approx(2.59 * env["diameter_m"])


def test_tit_bound_raised_to_1350():
    assert load_config().bounds["tit_K"].upper == 1350.0


def test_compressor_calibration_split_and_band():
    cal = yaml.safe_load(open("configs/components/kj66_compressor_calibration.yaml"))
    fit = cal["fit"]
    assert set(fit["calibration_points"]["speed_rpm"]).isdisjoint(
        fit["validation_points"]["speed_rpm"]
    )
    for k in ("delta_sigma", "diffuser_cf"):
        lo, hi = cal[k]["band_2sigma"]
        assert lo < cal[k]["value"] < hi

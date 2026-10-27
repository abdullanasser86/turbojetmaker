"""Round 5: compressor refit + extrapolation band (R1/R3), KJ-66 turbine from counts, Nickel 337 data, and the
physics-vs-missing-data classification."""

import math

import numpy as np
import pandas as pd
import pytest
import yaml

from turboml.analysis.feasibility import IN713C_SOLIDUS_K, classify_row
from turboml.data.generator import DatasetSpec, compressor_band, sample_designs
from turboml.physics.turbine_design import stage_from_counts

CAL = yaml.safe_load(open("configs/components/kj66_compressor_calibration.yaml", encoding="utf-8"))
CAL_V1 = yaml.safe_load(
    open("configs/components/kj66_compressor_calibration_v1_xiang90k.yaml", encoding="utf-8")
)


def test_refit_split_disjoint_and_top_line_held_out():
    fit = CAL["fit"]
    cal, val = set(fit["calibration_points"]["speed_rpm"]), set(
        fit["validation_points"]["speed_rpm"]
    )
    assert (
        cal.isdisjoint(val) and 120000 in val
    )  # the top measured line is a held-out (extrapolation) test
    for k in ("delta_sigma", "diffuser_cf"):
        lo, hi = CAL[k]["band_2sigma"]
        ilo, ihi = CAL[k]["band_inflated"]
        assert ilo < lo < CAL[k]["value"] < hi < ihi


def test_compressor_band_validated_vs_extrapolated():
    v = compressor_band(CAL, 400.0)
    assert v["tip_speed_regime"] == "validated" and v["scale"] == 1.0
    assert v["delta_sigma"] == tuple(CAL["delta_sigma"]["band_inflated"])
    e = compressor_band(CAL, 500.0)
    assert (
        e["tip_speed_regime"] == "extrapolated" and 2.5 < e["scale"] < 2.8
    )  # W(500) = 2.67 (growth law)
    # one-sided: lower slip bound and upper loss bound move, the others stay
    assert e["delta_sigma"][0] < v["delta_sigma"][0] and e["delta_sigma"][1] == v["delta_sigma"][1]
    assert e["diffuser_cf"][1] > v["diffuser_cf"][1] and e["diffuser_cf"][0] == v["diffuser_cf"][0]
    assert e["diffuser_cf"][0] > 0
    # W grows monotonically above the validated limit
    w = [compressor_band(CAL, u)["scale"] for u in (420, 440, 460, 480, 500)]
    assert all(b > a for a, b in zip(w, w[1:], strict=False))
    # the frozen v1 calibration has no extrapolation section: plain band, no regime
    assert compressor_band(CAL_V1, 480.0)["tip_speed_regime"] is None


def test_v5_sampling_flags_every_design():
    s = sample_designs(DatasetSpec.load("configs/dataset_phase4_v5.yaml"))
    for d in s:
        u2 = d["inputs"]["impeller_tip_speed_m_s"]
        assert d["tip_speed_regime"] == ("extrapolated" if u2 > 415.0 else "validated")
        lo, hi = d["compressor_band"]["delta_sigma"]
        if d["calibration_sampled"]:
            assert lo <= d["delta_sigma"] <= hi


def test_stage_from_counts_geometry():
    g = stage_from_counts(0.033, 0.7, 18, 24, 65.0, 60.0, 30.0, 0.01)
    r_hub, r_m, h = 0.7 * 0.033, 0.5 * (0.033 + 0.7 * 0.033), 0.033 * 0.3
    assert g.radius_hub_in == [pytest.approx(r_hub)] * 2 and g.radius_tip_out == [0.033, 0.033]
    assert g.pitch[0] == pytest.approx(2 * math.pi * r_m / 18) and g.pitch[1] == pytest.approx(
        2 * math.pi * r_m / 24
    )
    assert g.opening[0] == pytest.approx(g.pitch[0] * math.cos(math.radians(65.0)))
    assert g.tip_clearance == [0.0, pytest.approx(0.01 * h)]
    assert g.stagger_angle[1] == pytest.approx(0.5 * (30.0 - 60.0))


def test_ni337_anchors_and_ams5391():
    s = pd.read_csv("validation/cases/ni337_in713c_rupture_summary.csv", comment="#").set_index(
        ["T_F", "life_h"]
    )
    assert s.loc[(1350, 1000), "strength_ksi"] == 76 and s.loc[(1350, 10000), "strength_ksi"] == 56
    assert s.loc[(1350, 100), "strength_ksi"] == 97 and bool(s.loc[(1350, 100), "extrapolated"])
    t = pd.read_csv("validation/cases/ni337_in713c_rupture_tests.csv", comment="#")
    life = t[(t.T_F == 1800) & (t.stress_psi == 22000)].life_h
    assert (life >= 30).all()  # AMS 5391: >= 30 h at 1800 F / 22 ksi, as cast
    n393 = pd.read_csv("validation/cases/ni393_cast_rupture.csv", comment="#")
    m = n393[n393.alloy == "in713c"].merge(s.reset_index(), on=["T_F", "life_h"])
    assert len(m) >= 7 and np.allclose(
        m.strength_ksi_x, m.strength_ksi_y
    )  # same values where both tabulate


def _row(**kw):
    base = {"in_tit_K": 1200.0, "T03_K": 450.0, "active_constraints": "",
            "creep_in713c_rotor_status": "ok", "creep_in713c_ngv_status": "ok", "creep_in713c_rotor_metal_T_K": 1100.0,
            "creep_marm247_rotor_status": "ok", "creep_marm247_ngv_status": "ok", "creep_marm247_rotor_metal_T_K": 1100.0}  # fmt: skip
    return pd.Series({**base, **kw})


def test_classify_physics_vs_missing_data():
    an2 = classify_row(_row(creep_in713c_rotor_status="AN2", creep_marm247_rotor_status="AN2",
                            creep_in713c_ngv_status="above_material_data", creep_marm247_ngv_status="above_material_data"))  # fmt: skip
    assert an2["cause"] == "physics"
    data = classify_row(
        _row(creep_in713c_rotor_status="AN2", creep_marm247_ngv_status="above_material_data")
    )
    assert data["cause"] == "missing_data" and data["data_limited_materials"] == "marm247"
    # NGV hot streak above the IN-713C solidus is physics for IN-713C even though it is "above data"
    tit = 1300.0
    melt = classify_row(_row(in_tit_K=tit, T03_K=300.0, creep_in713c_ngv_status="above_material_data",
                             creep_marm247_rotor_status="AN2"))  # fmt: skip
    assert tit + 0.25 * (tit - 300.0) > IN713C_SOLIDUS_K and melt["cause"] == "physics"
    assert "ngv_above_solidus" in melt["in713c_physics"]


def test_kj66_turbine_labels():
    kj = yaml.safe_load(open("configs/components/kj66_turbine.yaml", encoding="utf-8"))
    for k in ("rotor_tip_radius_m", "n_ngv", "n_rotor", "tip_clearance_height_ratio"):
        assert kj[k]["kind"] == "published"
    for k in ("hub_tip", "ngv_gauging_deg", "rotor_gauging_deg"):
        assert kj[k]["kind"] == "calibrated, not measured" and kj[k]["value"] is not None
        lo, hi = kj[k]["range_profile"]
        assert lo <= kj[k]["value"] <= hi


def test_turbine_loss_multiplier_lowers_efficiency_and_defaults_to_identity():
    from turboml.gas import thermo
    from turboml.physics.turbine_meanline import AxialTurbineMeanline

    kj = yaml.safe_load(open("configs/components/kj66_turbine.yaml", encoding="utf-8"))
    g = stage_from_counts(0.033, kj["hub_tip"]["value"], 18, 24, kj["ngv_gauging_deg"]["value"],
                          kj["rotor_gauging_deg"]["value"], kj["rotor_le_deg"]["value"], 0.01)  # fmt: skip
    om = 100e3 * math.pi / 30
    base = AxialTurbineMeanline(g, thermo.products(0.015)).evaluate(820.0, 180e3, om, 1.8)
    same = AxialTurbineMeanline(g, thermo.products(0.015), loss_multiplier=1.0).evaluate(
        820.0, 180e3, om, 1.8
    )
    more = AxialTurbineMeanline(g, thermo.products(0.015), loss_multiplier=1.8).evaluate(
        820.0, 180e3, om, 1.8
    )
    assert base.converged and more.converged
    assert same.eta_tt == base.eta_tt  # bit-identical at k = 1
    assert more.eta_tt < base.eta_tt - 0.03


def test_v6_config_samples_k_loss_in_band_and_keeps_v5_identical():
    v6 = sample_designs(DatasetSpec.load("configs/dataset_phase4_v6.yaml"))
    tl = yaml.safe_load(open("configs/dataset_phase4_v6.yaml", encoding="utf-8"))[
        "turbine_loss_sampling"
    ]
    ks = np.array([d["turbine_loss_multiplier"] for d in v6])
    sampled = np.array([d["calibration_sampled"] for d in v6])
    assert np.all(ks[~sampled] == tl["value"])
    assert np.all((ks[sampled] >= tl["band"][0]) & (ks[sampled] <= tl["band"][1]))
    v5 = sample_designs(DatasetSpec.load("configs/dataset_phase4_v5.yaml"))
    assert all("turbine_loss_multiplier" not in d for d in v5)


def test_galvas_choking_incidence_table():
    import importlib.util

    s = importlib.util.spec_from_file_location("g", "scripts/phase4/galvas_r2_test.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    assert m.choking_incidence_deg(0.8, 50.0) == pytest.approx(
        1.5
    )  # all three tables cross at M = 0.8
    assert m.choking_incidence_deg(0.7, 45.0) == pytest.approx(
        -4.0
    )  # halfway between CINC40 (-5) and CINC50 (-3)
    assert math.isnan(m.choking_incidence_deg(0.4, 45.0))  # outside the tabulated Mach range


def test_engine_points_carry_retry_fields_and_eta_tt():
    from turboml.physics.engine import TURBINE_RETRY_PR_FACTORS, Engine, design_engine, load_engine_config

    cfg = load_engine_config()
    base = {k: (v["value"] if isinstance(v, dict) else v) for k, v in cfg["baseline_design"].items()}
    d = design_engine(base, cfg)
    dp = d.design_point
    assert dp.retries == 0 and dp.converged_on_retry is False
    assert 0.5 < dp.eta_t_tt < 1.0 and dp.eta_t_tt >= dp.eta_t_ts  # total-to-total >= total-to-static
    p = Engine(d, cfg).off_design(0.8 * dp.n_rpm)
    assert p.converged and p.retries == 0 and not p.converged_on_retry
    assert len(TURBINE_RETRY_PR_FACTORS) >= 2


def test_design_stage_retry_wrapper_reraises_non_convergence_reasons():
    from turboml.physics import turbine_design as td

    calls = []

    def fake(*a, pr0_factor=1.0, **k):
        calls.append(pr0_factor)
        raise ValueError("turbine design point failed: no_convergence (root finder did not converge)")

    orig = td._design_stage_once
    td._design_stage_once = fake
    try:
        with pytest.raises(ValueError, match="no_convergence"):
            td.design_stage(1.0, 1000.0, 3e5, 1e4, 1e5, 0.7, 1.5, 0.5)
    finally:
        td._design_stage_once = orig
    assert calls == [1.0, *td.DESIGN_RETRY_PR_FACTORS]  # first attempt + every restart, then the failure is raised

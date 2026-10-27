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

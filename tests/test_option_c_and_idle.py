"""Phase 4 v2: option C sizing (tip speed, turbine hub/tip), diagnostic efficiency hook, dataset v2 split."""

import numpy as np
import pandas as pd
import pytest

from turboml.data.generator import DatasetSpec, make_test_split, sample_designs
from turboml.physics import engine as E

CFG = E.load_engine_config()
BASE = {k: v["value"] if isinstance(v, dict) else v for k, v in CFG["baseline_design"].items()}


def test_option_c_recovers_baseline_speed_and_phi():
    """Prescribing the baseline's own tip speed and rotor hub/tip must give back its shaft speed and phi."""
    d0 = E.design_engine(BASE, CFG)
    tg = d0.turbine_geometry
    x = {k: v for k, v in BASE.items() if k not in ("n_rpm", "turb_phi")}
    x["impeller_tip_speed_m_s"] = d0.design_point.compressor_tip_speed_m_s
    x["turb_hub_tip"] = tg.radius_hub_out[1] / tg.radius_tip_out[1]
    d = E.design_engine(x, CFG)
    assert d.inputs["n_rpm"] == pytest.approx(BASE["n_rpm"], rel=1e-3)
    assert d.inputs["turb_phi"] == pytest.approx(BASE["turb_phi"], rel=1e-3)
    assert d.design_point.Fn_N == pytest.approx(d0.design_point.Fn_N, rel=1e-3)


def test_tip_speed_too_low_for_pr_is_labelled():
    ref = E._cached_reference(CFG)
    with pytest.raises(E.EngineFailure, match="design_compressor_tip_speed_too_low_for_pr"):
        E.size_compressor_tip_speed(ref, 0.5, 500.0, 4.0, 288.15, 101325.0 * 0.98)


def test_diag_hook_off_is_identical_and_on_moves_egt_the_right_way():
    d = E.design_engine(BASE, CFG)
    n = 0.35 * d.design_point.n_rpm
    p0 = E.Engine(d, CFG).off_design(n)
    e = E.Engine(d, CFG)
    e.diag_eta_delta = {"compressor": 0.0, "turbine": 0.0}
    p1 = e.off_design(n)
    assert p1.T05_K == pytest.approx(p0.T05_K, rel=1e-9) and p1.Fn_N == pytest.approx(
        p0.Fn_N, rel=1e-9
    )
    e = E.Engine(d, CFG)
    e.diag_eta_delta = {"compressor": 0.02}
    assert e.off_design(n).T05_K < p0.T05_K  # better compressor -> less turbine work -> cooler EGT
    assert p0.T05_static_K < p0.T05_K and 0 < p0.turbine_exit_mach < 1


def test_v2_split_frozen_and_samples_in_bounds():
    spec = DatasetSpec.load("configs/dataset_phase4_v2.yaml")
    m = make_test_split(2000, 0.15, spec.cfg["seed"])
    f = pd.read_csv("data/splits/phase4_v2_split.csv")
    assert np.array_equal(f.split.eq("test").to_numpy(), m)
    recs = sample_designs(spec)
    u2 = np.array([r["inputs"]["impeller_tip_speed_m_s"] for r in recs])
    ht = np.array([r["inputs"]["turb_hub_tip"] for r in recs])
    assert u2.min() >= 350 and u2.max() <= 500 and ht.min() >= 0.5 and ht.max() <= 0.95
    assert all("n_rpm" not in r["inputs"] and "turb_phi" not in r["inputs"] for r in recs)


def test_option_a_direct_sizing_recovers_kj66_and_builds_at_all_bounds():
    """At the KJ-66 reference point's own tip-speed specific speed, direct sizing gives back the KJ-66 wheel; every
    corner of the option-A box (W, U2, n_s,U) is a buildable wheel (user requirement 2026-10-25)."""
    import math

    ref = E._cached_reference(CFG)
    om = 110000 * math.pi / 30
    u2 = ref.geometry.radius_out_m * om
    ns = E.tip_speed_specific_speed(om, 0.235, 288.15, 101325.0, u2)
    assert ns == pytest.approx(0.6807, abs=1e-3)
    g, pt, n = E.size_compressor_direct(ref, 0.235, u2, ns, 288.15, 101325.0)
    assert n == pytest.approx(110000, rel=1e-9) and g.radius_out_m == pytest.approx(0.033, rel=1e-9)
    assert g.radius_tip_in_m == pytest.approx(ref.geometry.radius_tip_in_m, rel=1e-3)
    from turboml.config import load_config

    b = load_config().bounds
    for W in (b["air_mass_flow_kg_s"].lower, b["air_mass_flow_kg_s"].upper):
        for U in (b["impeller_tip_speed_m_s"].lower, b["impeller_tip_speed_m_s"].upper):
            for s in (b["comp_specific_speed"].lower, b["comp_specific_speed"].upper):
                g, pt, n = E.size_compressor_direct(ref, W, U, s, 288.15, 0.98 * 101325.0)
                assert pt.converged and pt.tip_speed_m_s == pytest.approx(U, rel=1e-9)


def test_v3_split_frozen_and_no_pr_variable():
    spec = DatasetSpec.load("configs/dataset_phase4_v3.yaml")
    f = pd.read_csv("data/splits/phase4_v3_split.csv")
    assert np.array_equal(
        f.split.eq("test").to_numpy(), make_test_split(2000, 0.15, spec.cfg["seed"])
    )
    r = sample_designs(spec)[0]["inputs"]
    assert "pr_c" not in r and "n_rpm" not in r and "comp_specific_speed" in r

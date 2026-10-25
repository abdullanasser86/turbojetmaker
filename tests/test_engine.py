"""Coupled engine model: design mode, off-design matching, labels, design card (slow: ~30 s)."""

import pytest

from turboml.physics.engine import (
    SAFETY_NOTE,
    Engine,
    design_card,
    design_engine,
    load_engine_config,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def baseline():
    cfg = load_engine_config()
    d = design_engine(cfg["baseline_design"], cfg)
    return cfg, d, Engine(d, cfg)


def test_design_point(baseline):
    cfg, d, _ = baseline
    p = d.design_point
    assert p.converged
    assert p.pr_c == pytest.approx(3.0, rel=1e-5)
    assert p.W_air_kg_s == pytest.approx(0.30)
    # turbine power balances the compressor (eta_mech)
    assert 0.99 * p.P_turb_W == pytest.approx(p.P_comp_W, rel=1e-5)


def test_off_design_reproduces_design(baseline):
    _, d, e = baseline
    q = e.off_design(d.design_point.n_rpm)
    assert q.converged
    assert q.Fn_N == pytest.approx(d.design_point.Fn_N, rel=1e-6)
    assert q.tit_K == pytest.approx(d.design_point.tit_K, rel=1e-6)


def test_part_speed_and_altitude(baseline):
    _, d, e = baseline
    q = e.off_design(0.8 * d.design_point.n_rpm)
    assert q.converged and q.Fn_N < d.design_point.Fn_N and q.tit_K < d.design_point.tit_K
    c = e.off_design(d.design_point.n_rpm, alt_m=3000.0, mach=0.5)
    assert c.converged and c.Fn_N < d.design_point.Fn_N


def test_failure_is_labelled(baseline):
    _, d, e = baseline
    q = e.off_design(0.2 * d.design_point.n_rpm)
    assert not q.converged and q.failure_reason


def test_design_card_has_safety_note(baseline):
    _, d, _ = baseline
    assert SAFETY_NOTE in design_card(d)

"""Centrifugal compressor meanline: verification, physics sanity, port check, regression."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from turboml.gas import thermo
from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def geom():
    return CentrifugalCompressorGeometry.from_yaml()


def test_loss_free_radial_impeller_is_ideal(geom):
    """No losses, sigma = 1, radial blades, frictionless diffuser -> Euler work U2^2 and eta = 1."""
    g = replace(geom, trailing_edge_angle_deg=0.0)
    c = CentrifugalCompressor(g, loss_multiplier=0.0, diffuser_cf=0.0)
    c.delta_sigma = 1.0 - c.slip_factor()  # -> sigma = 1
    p = c.evaluate(0.15, 90000)
    assert p.converged
    air = thermo.air()
    u2 = g.tip_speed_m_s(90000)
    assert air.h(p.T03_K) - air.h(288.15) == pytest.approx(u2**2, rel=1e-9)
    assert p.eta_tt == pytest.approx(1.0, abs=1e-9)
    assert p.pr_tt == pytest.approx(p.pr_tt_impeller, rel=1e-9)  # lossless diffuser


def test_physical_sanity(geom):
    c = CentrifugalCompressor(geom)
    p = c.evaluate(0.15, 90000)
    assert p.converged and 1.0 < p.pr_tt < p.pr_tt_impeller  # diffuser friction costs p0
    assert 0.5 < p.eta_tt < p.eta_tt_impeller < 1.0
    assert p.power_W == pytest.approx(
        0.15 * (thermo.air().h(p.T03_K) - thermo.air().h(288.15)), rel=1e-9
    )
    # more work from less slip; more friction -> lower efficiency
    assert CentrifugalCompressor(geom, delta_sigma=0.05).evaluate(0.15, 90000).pr_tt > p.pr_tt
    assert CentrifugalCompressor(geom, diffuser_cf=0.05).evaluate(0.15, 90000).eta_tt < p.eta_tt
    # PR rises with speed at fixed flow coefficient-ish flows
    assert c.evaluate(0.19, 110000).pr_tt > p.pr_tt


def test_failures_are_labelled(geom):
    c = CentrifugalCompressor(geom)
    p = c.evaluate(1.5, 90000)
    assert not p.converged and p.failure_reason.startswith("choke_")


def test_speed_line_flags(geom):
    d = CentrifugalCompressor(geom).speed_line(90000, np.linspace(0.05, 0.32, 12))
    assert {"surge_side", "surge_criterion", "is_peak_pr", "failure_reason"} <= set(d.columns)
    assert (~d.converged).any()  # high-flow end chokes; failures kept as rows
    if d.surge_side.any():
        assert (
            d.loc[d.surge_side, "mass_flow_kg_s"].max() < d.loc[d.converged, "mass_flow_kg_s"].max()
        )


@pytest.mark.slow
def test_port_matches_turboflow_impeller(geom):
    """With TurboFlow's loss convention and incidence reading, our impeller must reproduce
    TurboFlow's Oh-set impeller (which also uses our corrected Wiesner slip)."""
    from turboml.physics.compressor_turboflow import CentrifugalCompressorTF

    ours = CentrifugalCompressor(geom, loss_convention="turboflow", incidence_model="turboflow")
    tf = CentrifugalCompressorTF(geom)
    for N, m in [(70000, 0.11), (110000, 0.185)]:
        a, b = ours.evaluate(m, N), tf.evaluate(m, N)
        assert a.converged and b.converged
        assert a.pr_tt_impeller == pytest.approx(b.pr_tt, rel=5e-4)
        assert a.eta_tt_impeller == pytest.approx(b.eta_tt, abs=5e-4)


def test_turboflow_slip_hook_uses_degrees():
    from turboml.physics.compressor_turboflow import _install_slip_hook, wiesner_slip_velocity

    _install_slip_hook()
    from turboflow.centrifugal_compressor import slip_model as sm

    for beta in (0.0, -24.5, -40.0):
        v = sm.evaluate_slip(
            "wiesner", {"u_out": 400.0}, {"trailing_edge_angle": beta, "number_of_blades": 10}
        )
        assert np.isfinite(v)
        assert v == pytest.approx(400.0 * np.sqrt(np.cos(np.radians(beta))) / 10**0.7)
        assert v == pytest.approx(wiesner_slip_velocity(400.0, beta, 10))


def test_kj66_calibration_regression(geom):
    """Calibrated KJ-66 model (validation/phase2/kj66_summary.yaml) at 90 krpm vs measured mean."""
    s = yaml.safe_load((ROOT / "validation" / "phase2" / "kj66_summary.yaml").read_text())[
        "calibration"
    ]
    c = CentrifugalCompressor(geom, delta_sigma=s["delta_sigma"], diffuser_cf=s["diffuser_cf"])
    p = c.evaluate(0.148, 90000)
    assert p.pr_tt == pytest.approx(1.68, rel=0.03)  # measured cluster 1.66-1.695
    assert p.eta_tt == pytest.approx(0.755, abs=0.03)  # measured cluster 0.74-0.77

"""Physics sanity tests for the independent 0-D cycle model."""

from dataclasses import replace

import pytest

from turboml.gas import thermo as th
from turboml.physics.cycle_common import CycleInputs
from turboml.physics.cycle_simple import PerfectGas, design_point

LOSSLESS = dict(
    eta_c=1.0, eta_t=1.0, eta_mech=1.0, eta_burner=1.0, dp_burner_frac=0.0,
    inlet_recovery=1.0, dp_nozzle_frac=0.0, nozzle_cv=1.0,
)  # fmt: skip


@pytest.fixture
def base():
    return CycleInputs.from_yaml()


@pytest.mark.parametrize("pr", [1.5, 2.0, 2.4])
def test_ideal_brayton_air_standard(base, pr):
    """With a perfect gas, no losses and full expansion, the jet's kinetic power divided by
    the heat input must equal the closed-form Brayton efficiency 1 - PR^-(g-1)/g."""
    gas = PerfectGas(cp=1004.5, gamma=1.4)
    x = replace(base, pr_c=pr, tit_K=1000.0, **LOSSLESS)
    r = design_point(x, air_standard=True, gas=gas)
    assert r.converged and not r.choked  # convergent nozzle fully expands only if unchoked
    eta_th = 0.5 * x.W_air_kg_s * r.V9_m_s**2 / (r.Wf_kg_s * x.lhv_J_kg)
    assert eta_th == pytest.approx(1.0 - pr ** (-0.4 / 1.4), rel=1e-9)
    assert r.stations["3"].Tt_K == pytest.approx(288.15 * pr ** (0.4 / 1.4), rel=1e-12)


def test_energy_and_mass_balance(base):
    """Control volume around the whole engine (real-gas model, all losses on):
    W h_t0 + Wf h_fuel,eff = (W + Wf) h_t9 + mechanical loss + shaft off-take."""
    x = replace(base, power_offtake_W=500.0, mach=0.4, alt_m=2000.0)
    r = design_point(x)
    assert r.converged
    air, hot = th.air(), th.products(r.far)
    W, Wf = x.W_air_kg_s, r.Wf_kg_s
    lhs = W * air.h(r.stations["0"].Tt_K) + Wf * th.fuel_enthalpy_J_per_kg(x.lhv_J_kg, x.eta_burner)
    rhs = (W + Wf) * hot.h(r.stations["9"].Tt_K) + (1 - x.eta_mech) * r.P_turb_W + x.power_offtake_W
    assert lhs == pytest.approx(rhs, rel=1e-10)
    assert Wf / W == pytest.approx(r.far, rel=1e-14)
    # shaft balance
    assert x.eta_mech * r.P_turb_W == pytest.approx(r.P_comp_W + x.power_offtake_W, rel=1e-12)


def test_choked_nozzle_is_sonic(base):
    r = design_point(replace(base, pr_c=4.5, tit_K=1300.0, eta_c=0.78, eta_t=0.86))
    assert r.converged and r.choked and r.Ps9_Pa > 101325.0
    hot = th.products(r.far)
    Ts9 = hot.T_from_h(hot.h(r.stations["9"].Tt_K) - 0.5 * r.V9_m_s**2)
    assert r.V9_m_s == pytest.approx(hot.sound_speed(Ts9), rel=1e-8)


def test_thrust_scales_with_mass_flow(base):
    r1, r2 = design_point(base), design_point(replace(base, W_air_kg_s=2 * base.W_air_kg_s))
    assert r2.Fn_N == pytest.approx(2 * r1.Fn_N, rel=1e-12)
    assert r2.TSFC_kg_N_s == pytest.approx(r1.TSFC_kg_N_s, rel=1e-12)


@pytest.mark.parametrize(
    "field, better", [("eta_c", 0.80), ("eta_t", 0.88), ("eta_burner", 1.0), ("nozzle_cv", 1.0)]
)
def test_better_components_lower_tsfc(base, field, better):
    assert (
        design_point(replace(base, **{field: better})).TSFC_kg_N_s < design_point(base).TSFC_kg_N_s
    )


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tit_K": 400.0}, "constraint_tit_below_t3"),
        ({"eta_c": 0.4, "eta_t": 0.4}, "no_nozzle_pressure_ratio"),
    ],
)
def test_failures_are_labelled_not_raised(base, overrides, reason):
    r = design_point(replace(base, **overrides))
    assert not r.converged and r.failure_reason == reason

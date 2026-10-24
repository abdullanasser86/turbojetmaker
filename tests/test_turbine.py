"""Axial turbine (TurboFlow wrapper): Kofskey 1972 design point against measured data."""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def test_kofskey_design_point_vs_measured():
    from turboml.physics.turbine import AxialTurbine

    t = AxialTurbine()
    r = t.evaluate([t.operation_point(2.298, 1.0)]).iloc[0]
    assert r.converged
    data = pd.read_csv(
        ROOT / "validation" / "cases" / "kofskey1972_turbine_measured.csv", comment="#"
    )
    d100 = data[(data.speed_pct_design == 100)]

    def nearest(q):
        g = d100[d100.quantity == q]
        return g.iloc[(g.pressure_ratio_ts - 2.298).abs().argmin()].value

    assert r.mass_flow_kg_s == pytest.approx(nearest("mass_flow"), rel=0.03)
    assert 100 * r.eta_ts == pytest.approx(nearest("efficiency_ts"), abs=3.0)
    assert r.torque_N_m == pytest.approx(nearest("torque"), rel=0.06)


def test_failures_are_rows_not_exceptions():
    from turboml.physics.turbine import AxialTurbine

    t = AxialTurbine()
    d = t.evaluate([t.operation_point(2.0, 1.0), t.operation_point(50.0, 1.0)])
    assert len(d) == 2 and d.converged.iloc[0]
    assert "failure_reason" in d.columns

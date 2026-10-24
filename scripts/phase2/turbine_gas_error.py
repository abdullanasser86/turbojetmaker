"""Phase 2: error of treating turbine gas as air (TurboFlow/CoolProp) instead of combustion products.

For micro-turbojet turbine inlet conditions (TIT 1000-1250 K, FAR from the Phase 1 cycle,
PR_tt 1.8-2.6, eta 0.80) we compare, with TurboML's NASA-polynomial thermo:
  * isentropic and actual specific work for the same T0_in and pressure ratio,
  * the choked flow function  m sqrt(R T0)/(A p0) ~ sqrt(gamma) (2/(gamma+1))^((gamma+1)/(2(gamma-1)))
    expressed per sqrt(R) so it maps to mass flow at fixed T0, p0, area,
  * cp, gamma, R.
Also cross-checks TurboML air against CoolProp air (what TurboFlow uses) at the same states.

Usage:  uv run python scripts/phase2/turbine_gas_error.py   -> validation/phase2/turbine_gas_error.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from CoolProp.CoolProp import PropsSI

from turboml.gas import thermo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase2" / "turbine_gas_error.csv"


def flow_function(gas, T0: float) -> float:
    """Choked mass flux * sqrt(T0)/p0 [kg K^0.5 /(s m^2 Pa)] with gamma evaluated at T0."""
    g = gas.gamma(T0)
    return math.sqrt(g / gas.R) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))


def work(gas, T0: float, pr: float, eta: float) -> tuple[float, float]:
    h0 = gas.h(T0)
    w_is = h0 - gas.h(gas.T_isentropic(T0, 1.0 / pr))
    return w_is, eta * w_is


def main() -> int:
    air = thermo.air()
    rows = []
    for tit in (1000.0, 1100.0, 1250.0):
        for far in (0.015, 0.020, 0.025):
            prod = thermo.products(far)
            for pr in (1.8, 2.2, 2.6):
                wa_is, wa = work(air, tit, pr, 0.8)
                wp_is, wp = work(prod, tit, pr, 0.8)
                rows.append({
                    "TIT_K": tit, "FAR": far, "PR_tt": pr,
                    "work_products_J_kg": wp, "work_air_J_kg": wa,
                    "work_err_air_vs_products_%": 100 * (wa / wp - 1),
                    "flowfn_err_air_vs_products_%": 100 * (flow_function(air, tit) / flow_function(prod, tit) - 1),
                    "cp_err_%": 100 * (air.cp(tit) / prod.cp(tit) - 1),
                    "gamma_air": air.gamma(tit), "gamma_products": prod.gamma(tit),
                    "R_err_%": 100 * (air.R / prod.R - 1),
                })  # fmt: skip
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(
        d[
            [
                "TIT_K",
                "FAR",
                "PR_tt",
                "work_err_air_vs_products_%",
                "flowfn_err_air_vs_products_%",
                "cp_err_%",
            ]
        ]
        .round(
            {
                "FAR": 3,
                "work_err_air_vs_products_%": 2,
                "flowfn_err_air_vs_products_%": 2,
                "cp_err_%": 2,
            }
        )
        .to_string(index=False)
    )
    # CoolProp air vs TurboML air (cp and isentropic exponent)
    print("\nCoolProp 'air' vs TurboML air (cp at 1 bar):")
    for T in (300.0, 800.0, 1150.0):
        cp_cp = PropsSI("Cpmass", "T", T, "P", 1e5, "Air")
        print(
            f"  T={T:6.0f} K  cp CoolProp {cp_cp:8.2f}  TurboML {air.cp(T):8.2f}  diff {100 * (air.cp(T) / cp_cp - 1):+.2f} %"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

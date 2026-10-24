"""Phase 1 sanity check of the cycle model against JetCat datasheet values.

Two questions:

1. FORWARD: with our provisional baseline component assumptions
   (configs/cycle_baseline.yaml) and each engine's listed pressure ratio and mass flow,
   how far off are predicted thrust and SFC?
2. INVERSE: component efficiencies and TIT of these engines are not published. Thrust and
   SFC give two equations, so for an assumed compressor efficiency eta_c we can solve
   for the two unknowns (TIT, eta_t) that reproduce the datasheet exactly. If the implied
   values are physically plausible (EGT within the listed range, eta_t in a credible band),
   the cycle model is *consistent* with the data. This is a sanity check, not validation,
   because the component values are inferred, not measured.

Secondary losses (eta_mech, eta_burner, pressure losses, nozzle Cv) stay at the baseline
assumptions. Sensitivity to eta_burner is shown because it directly scales fuel flow.

Usage:  uv run python scripts/phase1/jetcat_inverse.py
Output: validation/phase1/jetcat_forward.csv, validation/phase1/jetcat_inverse.csv
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from turboml.physics.cycle_common import CycleInputs
from turboml.physics.cycle_simple import design_point

ROOT = Path(__file__).resolve().parents[2]
DB = yaml.safe_load((ROOT / "validation" / "cases" / "jetcat_datasheets.yaml").read_text())
OUT = ROOT / "validation" / "phase1"
KEROSENE_DENSITY_KG_L = 0.80  # assumption for the fuel-flow consistency check only
ETA_C_GRID = np.round(np.arange(0.60, 0.821, 0.02), 2)
ETA_B_VARIANTS = (0.90, 0.95, 1.00)


def engine_inputs(e: dict, base: CycleInputs) -> CycleInputs:
    return replace(base, W_air_kg_s=e["mass_flow_kg_s"], pr_c=e["pressure_ratio"])


def solve_inverse(e: dict, x0: CycleInputs) -> dict:
    """Find (TIT, eta_t) such that Fn and TSFC match the datasheet; eta_c is fixed in x0."""
    sfc_target = e["sfc_kg_N_h"] / 3600.0

    def resid(z):
        r = design_point(replace(x0, tit_K=z[0], eta_t=z[1]))
        if not r.converged:
            return [10.0, 10.0]
        return [r.Fn_N / e["thrust_N"] - 1.0, r.TSFC_kg_N_s / sfc_target - 1.0]

    sol = least_squares(resid, x0=[1150.0, 0.80], bounds=([800.0, 0.40], [1700.0, 1.0]), xtol=1e-12)
    r = design_point(replace(x0, tit_K=sol.x[0], eta_t=sol.x[1]))
    ok = bool(np.max(np.abs(sol.fun)) < 1e-6)
    return {
        "solved": ok,
        "tit_K": sol.x[0],
        "eta_t": sol.x[1],
        "egt_C": r.EGT_K - 273.15 if r.converged else np.nan,
        "choked": r.choked,
        "far": r.far,
        "max_resid": float(np.max(np.abs(sol.fun))),
    }


def main() -> int:
    base = CycleInputs.from_yaml()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Fuel-flow consistency (listed ml/min x 0.80 kg/L vs listed SFC x thrust):")
    for e in DB["engines"]:
        wf_listed = e["fuel_ml_min"] * KEROSENE_DENSITY_KG_L / 60.0 / 1000.0
        wf_sfc = e["sfc_kg_N_h"] * e["thrust_N"] / 3600.0
        print(f"  {e['name']:11s} ratio {wf_listed / wf_sfc:.3f}")

    # --- 1. forward
    rows = []
    for e in DB["engines"]:
        r = design_point(engine_inputs(e, base))
        rows.append(
            {
                "engine": e["name"],
                "thrust_datasheet_N": e["thrust_N"],
                "thrust_model_N": r.Fn_N,
                "thrust_err": r.Fn_N / e["thrust_N"] - 1,
                "sfc_datasheet": e["sfc_kg_N_h"],
                "sfc_model": r.TSFC_kg_N_s * 3600,
                "sfc_err": r.TSFC_kg_N_s * 3600 / e["sfc_kg_N_h"] - 1,
                "egt_model_C": r.EGT_K - 273.15,
                "egt_datasheet_max_C": e["egt_C"][1] if e["egt_C"] else np.nan,
            }
        )
    fwd = pd.DataFrame(rows)
    fwd.to_csv(OUT / "jetcat_forward.csv", index=False)
    print(
        f"\nFORWARD with baseline assumptions (eta_c={base.eta_c}, eta_t={base.eta_t}, "
        f"TIT={base.tit_K} K, eta_b={base.eta_burner}):"
    )
    print(fwd.round(3).to_string(index=False))

    # --- 2. inverse
    rows = []
    for e in DB["engines"]:
        for eta_b in ETA_B_VARIANTS:
            for eta_c in ETA_C_GRID:
                x0 = replace(engine_inputs(e, base), eta_c=float(eta_c), eta_burner=eta_b)
                rows.append(
                    {
                        "engine": e["name"],
                        "eta_burner": eta_b,
                        "eta_c": eta_c,
                        **solve_inverse(e, x0),
                    }
                )
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "jetcat_inverse.csv", index=False)
    print(
        f"\nINVERSE: implied (TIT, eta_t, EGT) reproducing datasheet thrust + SFC, eta_b={base.eta_burner}:"
    )
    sel = inv[(inv.eta_burner == base.eta_burner) & inv.eta_c.isin([0.64, 0.70, 0.76, 0.82])]
    print(
        sel.pivot(index="engine", columns="eta_c", values=["tit_K", "eta_t", "egt_C"])
        .round(3)
        .to_string()
    )
    print(f"\nunsolved combinations: {(~inv.solved).sum()} of {len(inv)}")
    print(
        "\nImplied EGT [C] vs combustion efficiency (eta_c = 0.70; EGT is almost independent of eta_c):"
    )
    print(
        inv[inv.eta_c == 0.70]
        .pivot(index="engine", columns="eta_burner", values="egt_C")
        .round(0)
        .to_string()
    )

    # --- 3. what combustion efficiency would make the datasheet EGT maximum consistent?
    print("\nCombustion efficiency needed for implied EGT = listed EGT maximum (eta_c = 0.70):")
    rows = []
    for e in DB["engines"]:
        if not e["egt_C"]:
            continue
        egt_max = e["egt_C"][1]

        def egt_minus_max(eta_b, e=e, egt_max=egt_max):
            x0 = replace(engine_inputs(e, base), eta_c=0.70, eta_burner=eta_b)
            return solve_inverse(e, x0)["egt_C"] - egt_max

        lo, hi = 0.5, 1.0
        if egt_minus_max(hi) < 0:
            need, note = 1.0, "consistent for every eta_b <= 1"
        elif egt_minus_max(lo) > 0:
            need, note = np.nan, "inconsistent even at eta_b = 0.5"
        else:
            note = ""
            from scipy.optimize import brentq

            need = brentq(egt_minus_max, lo, hi, xtol=1e-4)
            note = "EGT limit reached only if eta_b <= this"
        rows.append(
            {"engine": e["name"], "egt_max_C": egt_max, "eta_burner_max": need, "note": note}
        )
        print(f"  {e['name']:11s} EGT max {egt_max} C -> eta_b <= {need:.3f}  ({note})")
    pd.DataFrame(rows).to_csv(OUT / "jetcat_eta_b_needed.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

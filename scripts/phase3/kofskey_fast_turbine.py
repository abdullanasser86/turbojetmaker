"""Phase 3: fast axial-turbine meanline model on the Kofskey & Nusbaum (1972) single-stage turbine.

Two comparisons, both untuned:
  1. against the MEASURED data (validation/cases/kofskey1972_turbine_measured.csv), same method as the
     Phase 2 TurboFlow validation (cubic interpolation of each speed line to the measured pressure ratios);
  2. against TurboFlow's map (validation/phase2/kofskey_map.csv), point by point (port verification).
Both incidence-sign modes are run: "turboflow" (reproduces TurboFlow 0.1.18) and "rotor_plus" (TurboFlow's intended rotor sign).

Usage:  uv run python scripts/phase3/kofskey_fast_turbine.py
Output: validation/phase3/kofskey_fast_map.csv, kofskey_fast_comparison.csv, kofskey_fast_summary.yaml
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import CubicSpline

from turboml.physics.turbine import KOFSKEY_1972_CONFIG
from turboml.physics.turbine_meanline import AxialStageGeometry, AxialTurbineMeanline

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase3"
DATA = pd.read_csv(ROOT / "validation" / "cases" / "kofskey1972_turbine_measured.csv", comment="#")
TF_MAP = pd.read_csv(ROOT / "validation" / "phase2" / "kofskey_map.csv")
SPEEDS_PCT = [50, 70, 90, 100, 110]
PR_GRID = np.round(np.arange(1.6, 4.65, 0.05), 3)
T0, P0, OMEGA = 295.6, 1.38e5, 1627.0
MODEL_COL = {"mass_flow": "mass_flow_kg_s", "torque": "torque_N_m", "efficiency_ts": "eta_ts",
             "exit_flow_angle": "exit_flow_angle_deg"}  # fmt: skip


def run_map(mode: str) -> pd.DataFrame:
    turb = AxialTurbineMeanline(
        AxialStageGeometry.from_turboflow_yaml(KOFSKEY_1972_CONFIG), incidence_sign=mode
    )
    rows = []
    for sp in SPEEDS_PCT:
        for pr in PR_GRID:
            t0 = time.perf_counter()
            p = turb.evaluate(T0, P0, OMEGA * sp / 100, float(pr))
            rows.append({**p.scalars(), "speed_pct_design": sp, "incidence_sign": mode,
                         "time_ms": (time.perf_counter() - t0) * 1e3})  # fmt: skip
    return pd.DataFrame(rows)


def vs_measured(mp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in DATA[DATA.speed_pct_design.isin(SPEEDS_PCT)].itertuples():
        line = mp[(mp.speed_pct_design == r.speed_pct_design) & mp.converged].sort_values("pr_ts")
        col = MODEL_COL[r.quantity]
        inside = line.pr_ts.min() <= r.pressure_ratio_ts <= line.pr_ts.max()
        model = float(CubicSpline(line.pr_ts, line[col])(r.pressure_ratio_ts)) if inside else np.nan
        if r.quantity == "efficiency_ts":
            model *= 100.0
        rows.append({**r._asdict(), "model": model})
    c = pd.DataFrame(rows).drop(columns="Index")
    rel = c.quantity.isin(["mass_flow", "torque"])
    c["error"] = np.where(rel, c.model / c.value - 1, c.model - c.value)
    return c


def stats(c: pd.DataFrame) -> dict:
    out = {}
    for q, g in c.groupby("quantity"):
        e = g.error.dropna()
        scale = 100.0 if q in ("mass_flow", "torque") else 1.0
        unit = "%" if q in ("mass_flow", "torque") else ("pts" if q == "efficiency_ts" else "deg")
        out[q] = {f"mean_{unit}": round(float(e.mean() * scale), 2), f"mean_abs_{unit}": round(float(e.abs().mean() * scale), 2),
                  f"max_abs_{unit}": round(float(e.abs().max() * scale), 2), "n": int(e.size)}  # fmt: skip
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    maps = pd.concat([run_map("turboflow"), run_map("rotor_plus")], ignore_index=True)
    maps.drop(columns=["message"]).to_csv(OUT / "kofskey_fast_map.csv", index=False)
    summary = {"reference": "Kofskey & Nusbaum 1972, NASA TN D-6967 (measured, digitized by the TurboFlow authors)",
               "model": "turboml.physics.turbine_meanline.AxialTurbineMeanline (Benner losses, Aungier deviation, untuned)",
               "map_points_per_mode": int(len(maps) // 2)}  # fmt: skip
    comps = []
    for mode, mp in maps.groupby("incidence_sign"):
        c = vs_measured(mp).assign(incidence_sign=mode)
        comps.append(c)
        tf = TF_MAP[TF_MAP.converged].merge(
            mp[mp.converged], on=["speed_pct_design", "pr_ts"], suffixes=("_tf", "")
        )
        summary[mode] = {
            "converged": f"{int(mp.converged.sum())}/{len(mp)}",
            "time_ms_median_sweep": round(float(mp.time_ms.median()), 2),
            "vs_measured": stats(c),
            "vs_turboflow_map": {
                "n": int(len(tf)),
                "mass_flow_max_abs_pct": round(float((100 * (tf.mass_flow_kg_s / tf.mass_flow_kg_s_tf - 1)).abs().max()), 3),
                "eta_ts_max_abs_pts": round(float((100 * (tf.eta_ts - tf.eta_ts_tf)).abs().max()), 3),
                "power_max_abs_pct": round(float((100 * (tf.power_W / tf.power_W_tf - 1)).abs().max()), 3),
            },
        }  # fmt: skip
    pd.concat(comps).to_csv(OUT / "kofskey_fast_comparison.csv", index=False)
    (OUT / "kofskey_fast_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 2: axial turbine validation against Kofskey & Nusbaum (1972), NASA TN D-6967.

Model: turboml.physics.turbine.AxialTurbine (TurboFlow) with the published geometry
(third_party/turboflow_kofskey1972_1stage.yaml). No tuning.
Data:  validation/cases/kofskey1972_turbine_measured.csv (measured; digitized by TurboFlow's authors).

The model is evaluated on a PR grid (step 0.05) per measured speed line, in parallel, and
interpolated (cubic) to each measured pressure ratio. The grid is also the turbine map.

Usage:  uv run python scripts/phase2/kofskey_turbine_validation.py
Output: validation/phase2/kofskey_map.csv, kofskey_comparison.csv, kofskey_summary.yaml
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline

from turboml.physics.turbine import AxialTurbine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase2"
DATA = pd.read_csv(ROOT / "validation" / "cases" / "kofskey1972_turbine_measured.csv", comment="#")
SPEEDS_PCT = [50, 70, 90, 100, 110]  # 30 % excluded (as in TurboFlow's own validation)
PR_GRID = np.round(np.arange(1.6, 4.65, 0.05), 3)
MODEL_COL = {
    "mass_flow": "mass_flow_kg_s",
    "torque": "torque_N_m",
    "efficiency_ts": "eta_ts",
    "exit_flow_angle": "exit_flow_angle_deg",
}


def speed_line(speed_pct: int) -> pd.DataFrame:
    t = AxialTurbine()
    return t.map([speed_pct / 100.0], PR_GRID).assign(speed_pct_design=speed_pct)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mp = pd.concat(
        Parallel(n_jobs=-1)(delayed(speed_line)(s) for s in SPEEDS_PCT), ignore_index=True
    )
    mp.to_csv(OUT / "kofskey_map.csv", index=False)

    rows = []
    for r in DATA[DATA.speed_pct_design.isin(SPEEDS_PCT)].itertuples():
        line = mp[(mp.speed_pct_design == r.speed_pct_design) & mp.converged].sort_values("pr_ts")
        col = MODEL_COL[r.quantity]
        inside = line.pr_ts.min() <= r.pressure_ratio_ts <= line.pr_ts.max()
        model = (
            float(CubicSpline(line.pr_ts, line[col])(r.pressure_ratio_ts))
            if inside and len(line) > 3
            else np.nan
        )
        if r.quantity == "efficiency_ts":
            model *= 100.0  # data in %
        rows.append({**r._asdict(), "model": model})
    cmp_ = pd.DataFrame(rows).drop(columns="Index")
    rel = cmp_.quantity.isin(["mass_flow", "torque"])
    cmp_["error"] = np.where(rel, cmp_.model / cmp_.value - 1, cmp_.model - cmp_.value)
    cmp_["error_kind"] = np.where(rel, "relative", "absolute (pts or deg)")
    cmp_.to_csv(OUT / "kofskey_comparison.csv", index=False)

    summary = {"reference": "Kofskey & Nusbaum 1972, NASA TN D-6967 (measured, digitized by TurboFlow authors)",
               "model": "TurboFlow 0.1.18 axial turbine, Benner losses, Aungier deviation, no tuning",
               "map_points": int(len(mp)), "map_converged": int(mp.converged.sum()),
               "runtime_s": round(time.time() - t0, 1), "errors": {}}  # fmt: skip
    for q, g in cmp_.groupby("quantity"):
        e = g.error.dropna()
        scale = 100.0 if q in ("mass_flow", "torque") else 1.0
        unit = "%" if q in ("mass_flow", "torque") else ("pts" if q == "efficiency_ts" else "deg")
        summary["errors"][q] = {f"mean_{unit}": round(float(e.mean() * scale), 2),
                                f"mean_abs_{unit}": round(float(e.abs().mean() * scale), 2),
                                f"max_abs_{unit}": round(float(e.abs().max() * scale), 2),
                                "n": int(e.size), "n_outside_model_range": int(g.error.isna().sum())}  # fmt: skip
        by_speed = g.groupby("speed_pct_design").error.apply(
            lambda s, scale=scale: round(float(s.abs().mean() * scale), 2)
        )
        summary["errors"][q]["mean_abs_by_speed_pct"] = {int(k): v for k, v in by_speed.items()}
    (OUT / "kofskey_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

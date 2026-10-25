"""Widen the KJ-66 calibration band to include model-form error (user decision 2026-10-25, item 3).

The statistical 2-sigma band of (delta_sigma, diffuser_cf) (scripts/phase3/compressor_calibration_band.py) only
reflects the scatter of the 10 calibration rows. Here it is inflated by a common factor k:
    band(k) = centre +- k * (2-sigma half-width), for both parameters,
until at least 95 % of the HELD-OUT measured points (70/80/100/110 krpm; PR and efficiency rows) lie inside the
model's prediction envelope, i.e. between the min and max prediction over a 3 x 3 grid of the band at that point's
mass flow and speed. The measured values are compared without adding their digitization uncertainty (strict).

NOTE: this uses the held-out points to size the band. Their role as validation data for the POINT calibration is
unchanged (errors in docs/VALIDATION.md), but they are no longer independent evidence for the band width.

Usage:  uv run python scripts/phase4/compressor_band_inflation.py
Output: validation/phase4/compressor_band_inflation.csv / .yaml (values copied into the calibration config)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from turboml.compute import n_workers
from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"
EXP = pd.read_csv(ROOT / "validation" / "cases" / "kj66_compressor_exp_murray2009.csv", comment="#")
CAL = yaml.safe_load(
    (ROOT / "configs" / "components" / "kj66_compressor_calibration.yaml").read_text(
        encoding="utf-8"
    )
)
TARGET = 0.95
K_GRID = [1, 2, 2.25, 2.5, 2.75, 3, 4, 5, 6, 8, 10, 12, 15]


def predict(ds: float, cf: float, rows: pd.DataFrame) -> list[float]:
    geom = CentrifugalCompressorGeometry.from_yaml(
        ROOT / "configs" / "components" / "kj66_compressor.yaml"
    )
    c = CentrifugalCompressor(geom, delta_sigma=ds, diffuser_cf=cf)
    out = []
    for r in rows.itertuples():
        p = c.evaluate(r.mass_flow_kg_s, r.speed_rpm)
        if not p.converged:
            out.append(np.nan)
        else:
            out.append(p.pr_tt if r.quantity == "pressure_ratio" else p.eta_tt)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cal_speeds = CAL["fit"]["calibration_points"]["speed_rpm"]
    held = EXP[~EXP.speed_rpm.isin(cal_speeds)].reset_index(drop=True)
    c_ds, c_cf = CAL["delta_sigma"]["value"], CAL["diffuser_cf"]["value"]
    h_ds = 0.5 * (CAL["delta_sigma"]["band_2sigma"][1] - CAL["delta_sigma"]["band_2sigma"][0])
    h_cf = 0.5 * (CAL["diffuser_cf"]["band_2sigma"][1] - CAL["diffuser_cf"]["band_2sigma"][0])
    rows, chosen = [], None
    for k in K_GRID:
        grid = [
            (c_ds + a * k * h_ds, max(1e-4, c_cf + b * k * h_cf))
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
        ]
        preds = np.array(
            Parallel(n_jobs=n_workers())(delayed(predict)(ds, cf, held) for ds, cf in grid)
        )
        lo, hi = np.nanmin(preds, axis=0), np.nanmax(preds, axis=0)
        inside = (held.value >= lo) & (held.value <= hi)
        frac = float(inside.mean())
        by_q = {q: float(inside[held.quantity == q].mean()) for q in held.quantity.unique()}
        rows.append({"k": k, "delta_sigma_lo": c_ds - k * h_ds, "delta_sigma_hi": c_ds + k * h_ds,
                     "cf_lo": max(1e-4, c_cf - k * h_cf), "cf_hi": c_cf + k * h_cf, "coverage": frac, **{f"coverage_{q}": v for q, v in by_q.items()},
                     "n_failed_predictions": int(np.isnan(preds).sum())})  # fmt: skip
        print(rows[-1], flush=True)
        if frac >= TARGET:
            chosen = rows[-1]
            pd.DataFrame({"speed_rpm": held.speed_rpm, "mass_flow_kg_s": held.mass_flow_kg_s, "quantity": held.quantity,
                          "measured": held.value, "envelope_lo": lo, "envelope_hi": hi, "inside": inside}).to_csv(
                OUT / "compressor_band_envelope.csv", index=False)  # fmt: skip
            break
    pd.DataFrame(rows).to_csv(OUT / "compressor_band_inflation.csv", index=False)
    summary = {"target_coverage": TARGET, "held_out_points": int(len(held)), "k_grid_tried": [r["k"] for r in rows],
               "chosen": chosen, "statistical_2sigma_half_widths": {"delta_sigma": h_ds, "diffuser_cf": h_cf}}  # fmt: skip
    (OUT / "compressor_band_inflation.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

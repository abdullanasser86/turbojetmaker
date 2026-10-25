"""Uncertainty band of the KJ-66 compressor calibration factors (user decision 2026-10-25, item 1).

Re-fits delta_sigma and diffuser_cf on the CALIBRATION points only (90 krpm, Murray 2009 measured,
digitized; 5 PR + 5 efficiency rows) exactly as scripts/phase2/kj66_compressor_validation.py does, then
estimates the parameter covariance from the least-squares Jacobian:
    cov = s^2 (J^T J)^-1,   s^2 = RSS / (n - p).
The 2-sigma band is what Phase 4 samples. It is a STATISTICAL band from 10 digitized points: it does not
contain model-form error (the held-out speeds show PR errors up to 4.6 %). No validation point is used here.

Usage:  uv run python scripts/phase3/compressor_calibration_band.py
Output: printed; values are copied into configs/components/kj66_compressor_calibration.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry

ROOT = Path(__file__).resolve().parents[2]
EXP = pd.read_csv(ROOT / "validation" / "cases" / "kj66_compressor_exp_murray2009.csv", comment="#")
CALIB_SPEED = 90000


def main() -> int:
    geom = CentrifugalCompressorGeometry.from_yaml(
        ROOT / "configs" / "components" / "kj66_compressor.yaml"
    )
    cal = EXP[EXP.speed_rpm == CALIB_SPEED]

    def resid(z):
        c = CentrifugalCompressor(geom, delta_sigma=z[0], diffuser_cf=z[1])
        out = []
        for r in cal.itertuples():
            p = c.evaluate(r.mass_flow_kg_s, r.speed_rpm)
            if not p.converged:
                out.append(1.0)
            elif r.quantity == "pressure_ratio":
                out.append(p.pr_tt / r.value - 1)
            else:
                out.append((p.eta_tt - r.value) / r.value)
        return out

    sol = least_squares(
        resid, x0=[-0.1, 0.02], bounds=([-0.25, 0.002], [0.10, 0.5]), diff_step=1e-3
    )
    J, r = sol.jac, sol.fun
    n, p = len(r), len(sol.x)
    s2 = float(r @ r) / (n - p)
    cov = s2 * np.linalg.inv(J.T @ J)
    sd = np.sqrt(np.diag(cov))
    corr = cov[0, 1] / (sd[0] * sd[1])
    print("calibration rows:", n, "(speed", CALIB_SPEED, "rpm)")
    print(cal.to_string(index=False))
    print(
        f"delta_sigma = {sol.x[0]:.4f} +- {sd[0]:.4f} (1 sigma); 2-sigma band [{sol.x[0]-2*sd[0]:.4f}, {sol.x[0]+2*sd[0]:.4f}]"
    )
    print(
        f"diffuser_cf = {sol.x[1]:.4f} +- {sd[1]:.4f} (1 sigma); 2-sigma band [{sol.x[1]-2*sd[1]:.4f}, {sol.x[1]+2*sd[1]:.4f}]"
    )
    print(f"correlation {corr:.3f}; residual rms {np.sqrt(s2):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

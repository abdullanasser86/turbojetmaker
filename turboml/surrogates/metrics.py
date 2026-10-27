"""Accuracy and calibration metrics for the surrogates.

Regression (per target, in PHYSICAL units after inverting the transforms, except where noted):
* RMSE, MAE, R^2, and the relative error |y_hat/y - 1| (median, 95th percentile) for positive targets.
* Calibration of the predictive distribution (computed in the transformed, standardized space where the model is
  Gaussian): empirical coverage of the central p-intervals (a calibrated model covers ~p of the points), the
  standardized-residual std (~1 if calibrated), and the Gaussian negative log-likelihood.
Classification:
* accuracy, ROC AUC, Brier score (mean squared error of the probability), and the expected calibration error (ECE):
  the average gap between predicted probability and observed frequency over probability bins (Guo et al. 2017).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def regression(y: np.ndarray, y_hat: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(y_hat)
    y, y_hat = y[m], y_hat[m]
    if y.size == 0:
        return {"n": 0}
    err = y_hat - y
    ss = float(((y - y.mean()) ** 2).sum())
    out = {"n": int(y.size), "rmse": float(np.sqrt((err**2).mean())), "mae": float(np.abs(err).mean()),
           "r2": float(1 - (err**2).sum() / ss) if ss > 0 else float("nan")}  # fmt: skip
    if (y > 0).all():
        rel = np.abs(y_hat / y - 1)
        out.update(rel_err_median=float(np.median(rel)), rel_err_p95=float(np.quantile(rel, 0.95)))
    return out


def calibration(
    z: np.ndarray, mu: np.ndarray, sd: np.ndarray, levels=(0.5, 0.68, 0.9, 0.95)
) -> dict:
    m = np.isfinite(z) & np.isfinite(mu) & (sd > 0)
    z, mu, sd = z[m], mu[m], sd[m]
    if z.size == 0:
        return {"n": 0}
    r = (z - mu) / sd
    cov = {f"{lv:g}": float((np.abs(r) <= norm.ppf(0.5 + lv / 2)).mean()) for lv in levels}
    nll = float((0.5 * np.log(2 * np.pi * sd**2) + 0.5 * r**2).mean())
    return {"n": int(z.size), "coverage": cov, "std_of_standardized_residual": float(r.std()), "nll": nll,
            "miscalibration_area": float(np.mean([abs(cov[f'{lv:g}'] - lv) for lv in levels]))}  # fmt: skip


def classification(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    from sklearn.metrics import roc_auc_score

    y = y.astype(bool)
    out = {"n": int(y.size), "positives": int(y.sum()), "accuracy": float(((p >= 0.5) == y).mean()),
           "brier": float(((p - y) ** 2).mean())}  # fmt: skip
    out["roc_auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        s = idx == b
        if s.any():
            gap = abs(p[s].mean() - y[s].mean())
            ece += s.mean() * gap
            table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(s.sum()),
                          "mean_p": round(float(p[s].mean()), 3), "freq": round(float(y[s].mean()), 3)})  # fmt: skip
    out["ece"] = float(ece)
    out["reliability"] = table
    return out

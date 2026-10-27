"""Compressor refit R1 on Murray's MEASURED Chapter 5 speed lines, with held-out speed lines (user decision 2026-10-27).

Why
---
The stored calibration (delta_sigma, diffuser_cf) was fitted on the 90 krpm points of Murray's data as re-plotted by
Xiang et al. (2016). Against Murray's own Chapter 5 maps it is good from 50 to 100 krpm but over-predicts PR by +4 %
at 110 krpm and +7 % at 120 krpm (docs/VALIDATION.md, round 4). R1 refits the SAME two effective parameters on several
measured speed lines spread over the range, and keeps the other lines out of the fit to measure the error honestly.
No new loss term is added (R2 needs a published source first).

Model and data
--------------
* Model: KJ-66 meanline compressor (configs/components/kj66_compressor.yaml), standard-day inlet 288.15 K / 101.325 kPa,
  so model mass flow and speed are Murray's corrected values. Tip speed U2 = omega * r2, r2 = 33 mm.
* Data: validation/cases/murray2009/murray2009_ch5_maps_digitized.csv, Figs 5.12 (PR) and 5.14 (efficiency),
  digitized [MEAS]; corrected-speed classes 40-120 krpm.
* Split: CALIBRATION lines 50/70/90/110 krpm, HELD-OUT lines 40/60/80/100/120 krpm. The held-out set contains one
  interpolation test at every gap and two extrapolation tests (40 krpm below, 120 krpm above the calibrated range);
  the 120 krpm line is the top measured line (U2 = 415 m/s) and measures how the error grows at the high-speed edge.
* Residuals (same definition as the original fit, scripts/phase2/kj66_compressor_validation.py): PR/PR_meas - 1 at
  the measured corrected flow and eta/eta_meas - 1. Weighting: every calibration line contributes equally, and within
  a line the PR group and the efficiency group contribute equally (weight 1/sqrt(n_group)), so the 5-point efficiency
  clusters do not swamp the 1-2 PR points.
* Band: statistical cov = s^2 (J^T J)^-1, then the same model-form inflation rule as before (common factor k on the
  2-sigma half-widths, smallest k with >= 95 % of the HELD-OUT points of lines with U2 <= 415 m/s inside the 3 x 3 grid
  prediction envelope).
* Extrapolation (R3): the signed mean PR error per line vs tip speed is fitted with a quadratic in U2 (lines
  50-120 krpm); widening factor for U2 > U_top: W(U2) = e(U2) / e(U_top). See `growth_law` and the calibration file
  (`extrapolation`) for how W is applied.

Usage:  uv run python scripts/phase4/compressor_refit_murray.py
Output: validation/phase4/compressor_refit_murray.yaml, compressor_refit_murray_{pr,eta}.csv, compressor_refit_murray.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "validation" / "cases" / "murray2009"
OUT = ROOT / "validation" / "phase4"
MAPS = pd.read_csv(CASE / "murray2009_ch5_maps_digitized.csv", comment="#", dtype={"figure": str})
CAL_FILE = ROOT / "configs" / "components" / "kj66_compressor_calibration.yaml"
TREF, PREF = 288.15, 101325.0
CAL_LINES = (50, 70, 90, 110)
HELD_OUT = (40, 60, 80, 100, 120)
GEOM = CentrifugalCompressorGeometry.from_yaml()
U_TOP = GEOM.tip_speed_m_s(120e3)  # top measured speed line, 414.7 m/s
FLOWS = np.linspace(0.03, 0.30, 109)
K_GRID = (1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0)


def comp(ds: float, cf: float) -> CentrifugalCompressor:
    return CentrifugalCompressor(GEOM, delta_sigma=ds, diffuser_cf=cf, T0_in_K=TREF, p0_in_Pa=PREF)


PR_PTS = MAPS[MAPS.figure == "5.12"].reset_index(drop=True)
ETA_PTS = MAPS[MAPS.figure == "5.14"].reset_index(drop=True)


def residuals(p, lines=CAL_LINES) -> np.ndarray:
    c = comp(*p)
    out = []
    for n in lines:
        for pts, key in ((PR_PTS, "PR"), (ETA_PTS, "eta")):
            g = pts[pts.speed_class_krpm == n]
            w = 1.0 / math.sqrt(len(g))
            for r in g.itertuples():
                q = c.evaluate(r.Mcorr_kg_s, n * 1e3)
                model = (
                    (q.pr_tt if key == "PR" else q.eta_tt) if q.converged else 0.0
                )  # failure = large residual
                out.append(w * (model / getattr(r, key) - 1.0))
    return np.asarray(out)


def stats(e) -> dict:
    e = pd.Series(e).dropna()
    if e.empty:
        return {"n": 0}
    return {"mean": round(float(e.mean()), 3), "rms": round(float((e**2).mean() ** 0.5), 3),
            "max_abs": round(float(e.abs().max()), 3), "n": int(e.size)}  # fmt: skip


def evaluate(ds: float, cf: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Point errors on all lines (PR, closest-approach corrected flow, efficiency), as in murray_maps_validation.py."""
    c = comp(ds, cf)
    lines = {}
    for n in sorted(MAPS[MAPS.component == "compressor"].speed_class_krpm.unique()):
        sl = c.speed_line(n * 1e3, FLOWS)
        lines[int(n)] = sl[sl.converged][["mass_flow_kg_s", "pr_tt", "eta_tt"]].reset_index(
            drop=True
        )
    rows = []
    for r in PR_PTS.itertuples():
        sl = lines[int(r.speed_class_krpm)]
        p = c.evaluate(r.Mcorr_kg_s, r.speed_class_krpm * 1e3)
        u_m, u_pr = max(r.u_pub_Mcorr, r.u_dig_Mcorr_kg_s), max(r.u_pub_PR, r.u_dig_PR)
        mm = np.linspace(sl.mass_flow_kg_s.min(), sl.mass_flow_kg_s.max(), 2000)
        pp = np.interp(mm, sl.mass_flow_kg_s, sl.pr_tt)
        k = int(np.argmin(((mm - r.Mcorr_kg_s) / u_m) ** 2 + ((pp - r.PR) / u_pr) ** 2))
        rows.append({"speed_krpm": int(r.speed_class_krpm), "meas_Mcorr": r.Mcorr_kg_s, "meas_PR": r.PR, "u_PR": u_pr,
                     "model_PR": p.pr_tt if p.converged else np.nan, "model_Mcorr_closest": float(mm[k])})  # fmt: skip
    d = pd.DataFrame(rows)
    d["PR_err_pct"] = 100 * (d.model_PR / d.meas_PR - 1)
    d["Mcorr_err_pct"] = 100 * (d.model_Mcorr_closest / d.meas_Mcorr - 1)
    d["PR_within_unc"] = (d.model_PR - d.meas_PR).abs() <= d.u_PR
    erows = []
    for r in ETA_PTS.itertuples():
        p = c.evaluate(r.Mcorr_kg_s, r.speed_class_krpm * 1e3)
        erows.append({"speed_krpm": int(r.speed_class_krpm), "meas_Mcorr": r.Mcorr_kg_s, "meas_eta": r.eta,
                      "u_eta": max(r.u_pub_eta, r.u_dig_eta), "model_eta": p.eta_tt if p.converged else np.nan})  # fmt: skip
    e = pd.DataFrame(erows)
    e["eta_err_pts"] = 100 * (e.model_eta - e.meas_eta)
    e["eta_within_unc"] = (e.model_eta - e.meas_eta).abs() <= e.u_eta
    for df in (d, e):
        df["set"] = np.where(df.speed_krpm.isin(CAL_LINES), "calibration", "held_out")
        df["U2_m_s"] = [GEOM.tip_speed_m_s(n * 1e3) for n in df.speed_krpm]
    per = {}
    for n in sorted(d.speed_krpm.unique()):
        g, ge = d[d.speed_krpm == n], e[e.speed_krpm == n]
        per[int(n)] = {"set": "calibration" if n in CAL_LINES else "held_out",
                       "U2_m_s": round(GEOM.tip_speed_m_s(n * 1e3), 1),
                       "PR_err_pct": stats(g.PR_err_pct), "Mcorr_err_pct": stats(g.Mcorr_err_pct),
                       "eta_err_pts": stats(ge.eta_err_pts),
                       "PR_within_published_uncertainty": f"{int(g.PR_within_unc.sum())}/{len(g)}",
                       "eta_within_published_uncertainty": f"{int(ge.eta_within_unc.sum())}/{len(ge)}"}  # fmt: skip
    d.attrs["lines"] = lines
    return d, e, per


def envelope_coverage(centre, half, k, d_ho, e_ho) -> tuple[int, int]:
    """Count held-out points inside the min/max prediction over a 3 x 3 grid of the band centre +- k*half."""
    grid = [
        (centre[0] + a * k * half[0], centre[1] + b * k * half[1])
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
    ]
    prs = np.full((len(grid), len(d_ho)), np.nan)
    etas = np.full((len(grid), len(e_ho)), np.nan)
    for i, (ds, cf) in enumerate(grid):
        c = comp(ds, cf)
        for j, r in enumerate(d_ho.itertuples()):
            q = c.evaluate(r.meas_Mcorr, r.speed_krpm * 1e3)
            prs[i, j] = q.pr_tt if q.converged else np.nan
        for j, r in enumerate(e_ho.itertuples()):
            q = c.evaluate(r.meas_Mcorr, r.speed_krpm * 1e3)
            etas[i, j] = q.eta_tt if q.converged else np.nan
    inside = (
        (d_ho.meas_PR.values >= np.nanmin(prs, 0)) & (d_ho.meas_PR.values <= np.nanmax(prs, 0))
    ).sum()
    inside += (
        (e_ho.meas_eta.values >= np.nanmin(etas, 0)) & (e_ho.meas_eta.values <= np.nanmax(etas, 0))
    ).sum()
    return int(inside), len(d_ho) + len(e_ho)


def growth_law(d: pd.DataFrame, e: pd.DataFrame) -> dict:
    """Signed mean error per speed line vs tip speed, quadratic least squares in U2 over every line >= 50 krpm.

    The refit error is a smooth, convex function of U2 (about -1.5 % PR at 50-90 krpm, crossing zero near 100 krpm,
    +5 % at 120 krpm), so a quadratic in U2 is the lowest-order fit that follows it (residuals within the digitizing
    uncertainty); a straight line through the top three lines is reported as the milder alternative.
    40 krpm is excluded: a single marker on a nearly flat line, far below any design (U2 >= 350 m/s means >= 101 krpm
    on this wheel), and its efficiency error (+14 pts) is a separate low-speed issue.
    Widening factor for U2 > U_top: W(U2) = e(U2) / e(U_top), from the quadratic PR fit.
    """
    m = d.groupby("speed_krpm").agg(
        U2=("U2_m_s", "first"), pr=("PR_err_pct", "mean"), mc=("Mcorr_err_pct", "mean")
    )
    m["eta"] = e.groupby("speed_krpm").eta_err_pts.mean()
    m = m[m.index >= 50]
    u_eval = (414.7, 440.0, 460.0, 480.0, 500.0)
    out = {
        "form": "signed mean error per line = c2 U2^2 + c1 U2 + c0 (U2 in m/s), lines 50-120 krpm",
        "U_top_m_s": round(U_TOP, 1),
        "fits": {},
    }
    for col, unit in (("pr", "PR_err_pct"), ("mc", "Mcorr_err_pct"), ("eta", "eta_err_pts")):
        c = np.polyfit(m.U2, m[col], 2)
        out["fits"][unit] = {"coef_c2_c1_c0": [float(f"{v:.6g}") for v in c],
                             "residual_rms": round(float(np.sqrt(np.mean((m[col] - np.polyval(c, m.U2)) ** 2))), 3),
                             "extrapolated": {int(u): round(float(np.polyval(c, u)), 2) for u in u_eval}}  # fmt: skip
    c = np.polyfit(m.U2, m.pr, 2)
    lin = np.polyfit(m.U2[m.index >= 100], m.pr[m.index >= 100], 1)
    e_top = float(np.polyval(c, U_TOP))
    out["widening_factor_W"] = {int(u): round(float(np.polyval(c, u)) / e_top, 3) for u in u_eval}
    out["widening_factor_W_linear_top3_alternative"] = {
        int(u): round(float(np.polyval(lin, u) / np.polyval(lin, U_TOP)), 3) for u in u_eval
    }
    out["pr_coef_c2_c1_c0"] = [float(v) for v in c]
    out["lines"] = {int(n): {"U2_m_s": round(float(r.U2), 1), "PR_err_pct": round(r.pr, 3), "Mcorr_err_pct": round(r.mc, 3),
                             "eta_err_pts": round(float(r.eta), 3)} for n, r in m.iterrows()}  # fmt: skip
    return out


def plot(d, e, d_old) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = plt.cm.viridis(np.linspace(0, 0.9, 9))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for i, (n, sl) in enumerate(sorted(d.attrs["lines"].items())):
        ax.plot(
            sl.mass_flow_kg_s, sl.pr_tt, color=cols[i], lw=1, ls="-" if n in CAL_LINES else "--"
        )
        g = d[d.speed_krpm == n]
        ax.errorbar(g.meas_Mcorr, g.meas_PR, yerr=g.u_PR, fmt="o" if n in CAL_LINES else "s", color=cols[i], ms=4,
                    capsize=2, mfc=cols[i] if n in CAL_LINES else "white", label=f"{n} krpm ({'cal' if n in CAL_LINES else 'held out'})")  # fmt: skip
    ax.set(xlabel="corrected mass flow [kg/s]", ylabel="PR (tt)", xlim=(0.03, 0.26), ylim=(1.0, 2.6),
           title="Refit: model lines vs Murray Fig 5.12 (filled = calibration)")  # fmt: skip
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax = axes[1]
    for i, (n, sl) in enumerate(sorted(d.attrs["lines"].items())):
        ax.plot(
            sl.mass_flow_kg_s, sl.eta_tt, color=cols[i], lw=1, ls="-" if n in CAL_LINES else "--"
        )
        g = e[e.speed_krpm == n]
        ax.errorbar(g.meas_Mcorr, g.meas_eta, yerr=g.u_eta, fmt="o", color=cols[i], ms=3, capsize=1.5, alpha=0.8,
                    mfc=cols[i] if n in CAL_LINES else "white")  # fmt: skip
    ax.set(xlabel="corrected mass flow [kg/s]", ylabel="efficiency (tt)", xlim=(0.03, 0.26), ylim=(0.5, 0.9),
           title="Refit: efficiency vs Fig 5.14")  # fmt: skip
    ax = axes[2]
    for df, lab, mk in ((d_old, "stored calibration (90 krpm fit)", "x"), (d, "refit R1", "o")):
        m = df.groupby("speed_krpm").agg(U2=("U2_m_s", "first"), err=("PR_err_pct", "mean"))
        ax.plot(m.U2, m.err, mk + "-", label=lab)
    ax.axvline(U_TOP, color="k", lw=0.8, ls=":")
    ax.text(U_TOP + 2, ax.get_ylim()[0] + 0.3, "top measured line\n(validated limit)", fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel="impeller tip speed U2 [m/s]",
        ylabel="mean PR error per line [%]",
        title="PR error vs tip speed",
    )
    ax.legend(fontsize=7, frameon=False)
    for a in axes:
        a.grid(alpha=0.3)
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / "compressor_refit_murray.png", dpi=140)
    plt.close(fig)


def main() -> int:
    old = yaml.safe_load(CAL_FILE.read_text(encoding="utf-8"))
    p_old = (old["delta_sigma"]["value"], old["diffuser_cf"]["value"])
    fit = least_squares(
        residuals, p_old, bounds=([-0.25, 0.002], [0.10, 0.5]), x_scale=[0.01, 0.01], diff_step=1e-3
    )
    n, p = fit.fun.size, 2
    s2 = float(fit.fun @ fit.fun) / (n - p)
    cov = s2 * np.linalg.inv(fit.jac.T @ fit.jac)
    sig = np.sqrt(np.diag(cov))
    corr = float(cov[0, 1] / (sig[0] * sig[1]))
    ds, cf = (float(v) for v in fit.x)
    d, e, per = evaluate(ds, cf)
    d_old, e_old, per_old = evaluate(*p_old)
    ho_d = d[(d.set == "held_out") & (d.U2_m_s <= U_TOP + 0.5)]
    ho_e = e[(e.set == "held_out") & (e.U2_m_s <= U_TOP + 0.5)]
    half = (2 * sig[0], 2 * sig[1])
    cover = {}
    k_sel = None
    for k in K_GRID:
        ins, tot = envelope_coverage((ds, cf), half, k, ho_d, ho_e)
        cover[k] = f"{ins}/{tot}"
        if ins / tot >= 0.95:
            k_sel = k
            break
    growth = growth_law(d, e)
    ho = {"PR_err_pct": stats(d[d.set == "held_out"].PR_err_pct),
          "Mcorr_err_pct": stats(d[d.set == "held_out"].Mcorr_err_pct),
          "eta_err_pts": stats(e[e.set == "held_out"].eta_err_pts),
          "PR_within_published_uncertainty": f"{int(d[d.set == 'held_out'].PR_within_unc.sum())}/{int((d.set == 'held_out').sum())}",
          "eta_within_published_uncertainty": f"{int(e[e.set == 'held_out'].eta_within_unc.sum())}/{int((e.set == 'held_out').sum())}"}  # fmt: skip
    res = {
        "data": "Murray 2009 Figs 5.12/5.14, digitized [MEAS] (validation/cases/murray2009/murray2009_ch5_maps_digitized.csv)",
        "split": {"calibration_lines_krpm": list(CAL_LINES), "held_out_lines_krpm": list(HELD_OUT)},
        "refit": {"delta_sigma": round(ds, 4), "diffuser_cf": round(cf, 4), "sigma_1": [round(float(sig[0]), 4), round(float(sig[1]), 4)],
                  "parameter_correlation": round(corr, 3), "n_residuals": int(n), "residual_rms_weighted": round(math.sqrt(s2), 4),
                  "slip_factor_on_kj66": round(comp(ds, cf).slip_factor(), 4), "optimizer_status": int(fit.status)},
        "previous": {"delta_sigma": p_old[0], "diffuser_cf": p_old[1]},
        "band_inflation": {"rule": ">= 95 % of held-out points with U2 <= U_top inside the 3x3 envelope", "coverage_by_k": cover, "k": k_sel},
        "per_speed_line_refit": per,
        "per_speed_line_previous_calibration": per_old,
        "held_out_all_lines_refit": ho,
        "held_out_all_lines_previous_calibration": {"PR_err_pct": stats(d_old[d_old.set == "held_out"].PR_err_pct),
                                                     "Mcorr_err_pct": stats(d_old[d_old.set == "held_out"].Mcorr_err_pct),
                                                     "eta_err_pts": stats(e_old[e_old.set == "held_out"].eta_err_pts)},
        "error_growth_vs_tip_speed": growth,
    }  # fmt: skip
    d.to_csv(OUT / "compressor_refit_murray_pr.csv", index=False)
    e.to_csv(OUT / "compressor_refit_murray_eta.csv", index=False)
    plot(d, e, d_old)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "compressor_refit_murray.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

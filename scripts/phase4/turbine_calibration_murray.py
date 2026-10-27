"""KJ-66 turbine rebuilt from published counts, with hub/tip and blade angles CALIBRATED on Murray's measured turbine
maps (user decision 2026-10-27, round 5 item 2). Values produced here are "calibrated, not measured".

Published geometry (configs/components/kj66_turbine.yaml)
---------------------------------------------------------
Xu, Wu, Liu & Piao, GPPS-TC-2021-0117 (CC BY 4.0), "New KJ66" micro turbine engine: rotor diameter 66 mm, 18 NGVs,
24 rotor blades, tip clearance 1 % of blade height. CAUTION (user): a redesigned one-spool KJ66 derived from
Teixeira et al. (GT2018), not certainly the turbine of Murray's ArtesJet KJ-66 -- "a published KJ66 variant".
NOT published: hub radius / blade height and the blade/NGV angles. These three are calibrated here:
  hub_tip            rotor (= NGV) hub/tip ratio, cylindrical annulus (assumption),
  ngv_gauging_deg    NGV throat (gauging) angle, cos g = o/s, from axial -- sets the NGV exit flow angle,
  rotor_gauging_deg  rotor throat angle (unsigned), sets the rotor exit relative flow angle.
Rotor metal inlet angle: zero incidence at Murray's top measured point (Appendix D, D.10, 120 krpm) -- a design rule,
not a fitted value. Construction rules: turboml.physics.turbine_design.stage_from_counts.

Data and split
--------------
Murray Figs 5.16 (corrected AIR mass flow vs total-to-total PR) and 5.18 (total-to-total efficiency vs PR), digitized
[MEAS] (validation/cases/murray2009/murray2009_ch5_maps_digitized.csv). Murray's turbine efficiency is DERIVED from
the compressor work (published uncertainty +-0.08 at the top line to +-0.34 at 26 krpm).
CALIBRATION lines (corrected speed, krpm): 26, 38, 50, 62. HELD-OUT: 20, 32, 44, 56, 68 (68 = top line, extrapolation
above the calibrated range; 20 = below it, flow only).
Each point is evaluated at its corrected speed with the turbine-inlet state of Appendix D interpolated in turbine
corrected speed (T04, P04, FAR; clamped below 28 krpm, the lowest Appendix D point), at the measured PR_tt.
Murray Appendix D turbine-exit mid-span velocity V5 (CALC_MEAS from P05 and Ps5, u = 18-45 m/s), evaluated at each
D-point's own T04, P04, FAR, speed and PR_tt: model rotor-exit absolute velocity vs V5. ASSUMPTION: station 5 is the
rotor exit plane of the cylindrical annulus (V5 then constrains the exit annulus area, i.e. the hub/tip ratio).
D-points: CALIBRATION D.1, D.3, ..., D.15 (odd), HELD-OUT D.2, ..., D.14 (even).

Two fits (both reported):
  F  (adopted): flow (Fig 5.16) + exit velocity (V5). Efficiency is NOT fitted: it is held out entirely and reported.
  FE (as first specified): flow + efficiency (Fig 5.18). Reported to show what the efficiency data do to the geometry.
  Q  flow only: shows that the flow map alone does not identify hub/tip (only the NGV throat area).
Residuals: relative (model/meas - 1); equal weight per calibration speed line and per group within a line; the V5
group gets the same total weight as the flow group. Identifiability: profile of fit F over a hub/tip grid.

Usage:  uv run python scripts/phase4/turbine_calibration_murray.py
Output: validation/phase4/turbine_calibration_murray.yaml/.csv/.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq, least_squares

from turboml.gas import thermo
from turboml.physics.turbine_design import stage_from_counts
from turboml.physics.turbine_meanline import AxialTurbineMeanline

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "validation" / "cases" / "murray2009"
OUT = ROOT / "validation" / "phase4"
MAPS = pd.read_csv(CASE / "murray2009_ch5_maps_digitized.csv", comment="#", dtype={"figure": str})
AD = pd.read_csv(CASE / "murray2009_appD_running_data.csv", comment="#").sort_values(
    "turb_Ncorr_krpm"
)
KJ = yaml.safe_load(
    (ROOT / "configs" / "components" / "kj66_turbine.yaml").read_text(encoding="utf-8")
)
TREF, PREF = 288.15, 101325.0
CAL_LINES = (26, 38, 50, 62)
HELD_OUT = (20, 32, 44, 56, 68)
NAMES = ("hub_tip", "ngv_gauging_deg", "rotor_gauging_deg")
X0 = (0.60, 70.0, 60.0)
LB, UB = (0.40, 50.0, 35.0), (0.90, 82.0, 78.0)
PTS = MAPS[MAPS.component == "turbine"].reset_index(drop=True)
TOP = AD[AD.figure == "D.10"].iloc[0]
D_CAL = {f"D.{i}" for i in range(1, 16, 2)}


def inlet_state(ncorr_krpm: float) -> tuple[float, float, float]:
    """(T04, P04, FAR) of Appendix D interpolated in turbine corrected speed."""
    x = AD.turb_Ncorr_krpm
    return (float(np.interp(ncorr_krpm, x, AD.T04_K)), float(np.interp(ncorr_krpm, x, AD.P04_kPa)) * 1e3,
            float(np.interp(ncorr_krpm, x, AD.FAR)))  # fmt: skip


def at_pr_tt(t: AxialTurbineMeanline, T04, P04, omega, pr_tt):
    def f(pr_ts):
        q = t.evaluate(T04, P04, omega, pr_ts)
        return (P04 / q.p0_out_Pa - pr_tt) if q.converged else float("nan")

    try:
        pr_ts = brentq(f, pr_tt * 1.0005, pr_tt * 1.8, xtol=1e-7)
    except ValueError:
        return None
    q = t.evaluate(T04, P04, omega, pr_ts)
    return q if q.converged else None


def build(p, n_le_iter: int = 3):
    """Geometry for parameters p; rotor LE angle set to the relative inflow angle at the D.10 point (zero incidence)."""
    hub_tip, g_ngv, g_rot = p
    le = 30.0
    gas = thermo.products(float(TOP.FAR))
    om = TOP.speed_krpm * 1e3 * math.pi / 30
    for _ in range(n_le_iter):
        g = stage_from_counts(KJ["rotor_tip_radius_m"]["value"], hub_tip, KJ["n_ngv"]["value"], KJ["n_rotor"]["value"],
                              g_ngv, g_rot, le, KJ["tip_clearance_height_ratio"]["value"])  # fmt: skip
        q = at_pr_tt(AxialTurbineMeanline(g, gas), TOP.T04_K, TOP.P04_kPa * 1e3, om, TOP.turb_PR)
        if q is None or not math.isfinite(q.rotor_inlet_beta_deg):
            return g, le
        le = float(q.rotor_inlet_beta_deg)
    g = stage_from_counts(KJ["rotor_tip_radius_m"]["value"], hub_tip, KJ["n_ngv"]["value"], KJ["n_rotor"]["value"],
                          g_ngv, g_rot, le, KJ["tip_clearance_height_ratio"]["value"])  # fmt: skip
    return g, le


def predict(p, k_loss: float = 1.0, geometry=None) -> pd.DataFrame:
    """Model vs Figs 5.16/5.18 for parameters p (or a fixed `geometry` = (AxialStageGeometry, rotor LE)), with the
    turbine loss multiplier k_loss (D-064)."""
    g, le = build(p) if geometry is None else geometry
    rows = []
    turbs = {}
    for r in PTS.itertuples():
        n = r.speed_class_krpm
        T04, P04, far = inlet_state(n)
        if far not in turbs:
            turbs[far] = AxialTurbineMeanline(g, thermo.products(far), loss_multiplier=k_loss)
        om = n * 1e3 * math.sqrt(T04 / TREF) * math.pi / 30
        q = at_pr_tt(turbs[far], T04, P04, om, r.PR)
        row = {"figure": r.figure, "speed_krpm": int(n), "meas_PR": r.PR,
               "meas": r.Mcorr_kg_s if r.figure == "5.16" else r.eta,
               "u_pub": r.u_pub_Mcorr if r.figure == "5.16" else r.u_pub_eta,
               "u_dig": r.u_dig_Mcorr_kg_s if r.figure == "5.16" else r.u_dig_eta, "model": np.nan, "rotor_le_deg": le}  # fmt: skip
        if q is not None:
            m_air = q.mass_flow_kg_s / (
                1 + far
            )  # Murray's turbine corrected flow uses the AIR mass flow
            row["model"] = (
                m_air * math.sqrt(T04 / TREF) * PREF / P04 if r.figure == "5.16" else q.eta_tt
            )
            row.update(choked=q.choked, rotor_incidence_deg=q.rotor_incidence_deg, ngv_exit_mach=q.stator_exit_mach,
                       exit_flow_angle_deg=q.exit_flow_angle_deg)  # fmt: skip
        rows.append(row)
    d = pd.DataFrame(rows)
    d["set"] = np.where(d.speed_krpm.isin(CAL_LINES), "calibration", "held_out")
    d["err"] = np.where(d.figure == "5.16", 100 * (d.model / d.meas - 1), 100 * (d.model - d.meas))
    d["within_pub"] = (d.model - d.meas).abs() <= np.maximum(d.u_pub, d.u_dig)
    return d


def predict_v5(p) -> pd.DataFrame:
    """Model rotor-exit absolute velocity at every Appendix D point vs Murray's V5."""
    g, _le = build(p)
    rows = []
    for r in AD.itertuples():
        gas = thermo.products(float(r.FAR))
        q = at_pr_tt(
            AxialTurbineMeanline(g, gas),
            r.T04_K,
            r.P04_kPa * 1e3,
            r.speed_krpm * 1e3 * math.pi / 30,
            r.turb_PR,
        )
        v = np.nan
        if q is not None:
            v = q.exit_mach * math.sqrt(gas.gamma(q.T_out_K) * gas.R * q.T_out_K)
        rows.append({"figure": r.figure, "speed_krpm": r.speed_krpm, "turb_Ncorr_krpm": r.turb_Ncorr_krpm,
                     "meas_V5": r.V5_m_s, "u_V5": r.u_V5_m_s, "model_V5": v,
                     "set": "calibration" if r.figure in D_CAL else "held_out"})  # fmt: skip
    d = pd.DataFrame(rows)
    d["err_pct"] = 100 * (d.model_V5 / d.meas_V5 - 1)
    d["within_pub"] = (d.model_V5 - d.meas_V5).abs() <= d.u_V5
    return d


def residuals(p, groups=("5.16", "V5")) -> np.ndarray:
    d = predict(p)
    d = d[(d.set == "calibration") & d.figure.isin(groups)]
    out = []
    for (_n, _fig), g in d.groupby(["speed_krpm", "figure"]):
        w = 1.0 / math.sqrt(len(g))
        out.extend(
            w * np.nan_to_num(g.model / g.meas - 1.0, nan=1.0)
        )  # failed point = large residual
    n_lines = d[d.figure == "5.16"].speed_krpm.nunique()
    if "V5" in groups:
        v = predict_v5(p)
        v = v[v.set == "calibration"]
        out.extend(
            math.sqrt(n_lines / len(v)) * np.nan_to_num(v.model_V5 / v.meas_V5 - 1.0, nan=1.0)
        )
    return np.asarray(out)


def run_fit(groups, x0=X0, lb=LB, ub=UB, names=NAMES, fixed=None) -> dict:
    """Least-squares fit of the free parameters (`fixed` = {index: value} holds some at a value)."""
    fixed = fixed or {}
    free = [i for i in range(len(x0)) if i not in fixed]

    def full(z):
        p = list(x0)
        for i, v in fixed.items():
            p[i] = v
        for i, v in zip(free, z, strict=True):
            p[i] = v
        return p

    fit = least_squares(lambda z: residuals(full(z), groups), [x0[i] for i in free], bounds=([lb[i] for i in free], [ub[i] for i in free]),
                        x_scale=[[0.05, 2.0, 2.0][i] for i in free], diff_step=[2e-3] * len(free))  # fmt: skip
    n, k = fit.fun.size, len(free)
    s2 = float(fit.fun @ fit.fun) / (n - k)
    jtj = fit.jac.T @ fit.jac
    cov = s2 * np.linalg.pinv(jtj)
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(sig, sig)
    p = full(fit.x)
    return {"p": p, "rss": float(fit.fun @ fit.fun), "s2": s2, "n": n, "status": int(fit.status),
            "cond": float(np.linalg.cond(jtj)),
            "calibrated": {names[i]: {"value": round(float(p[i]), 4 if i == 0 else 2), "sigma_1": round(float(sig[j]), 4 if i == 0 else 2),
                                      "at_bound": bool(min(abs(p[i] - lb[i]), abs(p[i] - ub[i])) < 1e-4 * max(1.0, abs(ub[i])))}
                           for j, i in enumerate(free)},
            "correlation": {f"{names[free[a]]}~{names[free[b]]}": round(float(corr[a, b]), 3) for a in range(k) for b in range(a + 1, k)}}  # fmt: skip


def stats(e) -> dict:
    e = pd.Series(e).dropna()
    if e.empty:
        return {"n": 0}
    return {"mean": round(float(e.mean()), 2), "rms": round(float((e**2).mean() ** 0.5), 2),
            "max_abs": round(float(e.abs().max()), 2), "n": int(e.size)}  # fmt: skip


def per_line(d) -> dict:
    out = {}
    for n, g in d.groupby("speed_krpm"):
        f, e = g[g.figure == "5.16"], g[g.figure == "5.18"]
        out[int(n)] = {"set": g.set.iloc[0], "Mcorr_err_pct": stats(f.err), "eta_err_pts": stats(e.err),
                       "Mcorr_within_published_uncertainty": f"{int(f.within_pub.sum())}/{len(f)}",
                       "eta_within_published_uncertainty": f"{int(e.within_pub.sum())}/{len(e)}"}  # fmt: skip
    return out


def plot(d, dfe, d0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, figno, lab in (
        (axes[0], "5.16", "corrected air mass flow [kg/s]"),
        (axes[1], "5.18", "efficiency (tt)"),
    ):
        g = d[d.figure == figno].sort_values("meas_PR")
        for s, mk in (("calibration", "o"), ("held_out", "s")):
            gg = g[g.set == s]
            ax.errorbar(gg.meas_PR, gg.meas, yerr=np.maximum(gg.u_pub, gg.u_dig) if figno == "5.16" else None,
                        fmt=mk, color="k", mfc="k" if s == "calibration" else "white", ms=4, capsize=2,
                        label=f"Murray Fig {figno} ({'cal' if s == 'calibration' else 'held out'})")  # fmt: skip
        ax.plot(g.meas_PR, g.model, "-", color="#2a78d6", label="fit F (flow + V5), adopted")
        ge = dfe[dfe.figure == figno].sort_values("meas_PR")
        ax.plot(ge.meas_PR, ge.model, "--", color="#7a3ad6", label="fit FE (flow + efficiency)")
        g0 = d0[d0.figure == figno].sort_values("meas_PR")
        ax.plot(g0.meas_PR, g0.model, ":", color="#d65a2a", label="initial guess")
        ax.set(xlabel="turbine PR (tt)", ylabel=lab, title=f"Turbine vs Murray Fig {figno}")
        ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    for figno, mk, lab in (
        ("5.16", "o-", "flow error [%]"),
        ("5.18", "s-", "efficiency error [pts]"),
    ):
        m = d[d.figure == figno].groupby("speed_krpm").err.mean()
        ax.plot(m.index, m.values, mk, label=f"fit F: {lab}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel="turbine corrected speed [krpm]",
        ylabel="model - Murray",
        title="Error per speed line",
    )
    ax.legend(fontsize=7, frameon=False)
    for a in axes:
        a.grid(alpha=0.3)
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / "turbine_calibration_murray.png", dpi=140)
    plt.close(fig)


def summarize(fr: dict, label: str) -> dict:
    p = fr["p"]
    d, v = predict(p), predict_v5(p)
    g, le = build(p)
    gas = thermo.products(float(TOP.FAR))
    q = at_pr_tt(
        AxialTurbineMeanline(g, gas),
        TOP.T04_K,
        TOP.P04_kPa * 1e3,
        TOP.speed_krpm * 1e3 * math.pi / 30,
        TOP.turb_PR,
    )
    low = d[d.speed_krpm <= 44]
    r_t = KJ["rotor_tip_radius_m"]["value"]
    out = {
        "fit": label, "calibrated": fr["calibrated"], "correlation": fr["correlation"],
        "jtj_condition_number": float(f"{fr['cond']:.3g}"), "n_residuals": fr["n"],
        "residual_rms_weighted": round(math.sqrt(fr["s2"]), 4), "optimizer_status": fr["status"],
        "rotor_le_deg_zero_incidence_at_D10": round(le, 2),
        "derived_geometry": {"r_hub_m": round(p[0] * r_t, 5), "blade_height_m": round((1 - p[0]) * r_t, 5),
                             "tip_clearance_m": round(g.tip_clearance[1], 6), "ngv_chord_m": round(g.chord[0], 5),
                             "rotor_chord_m": round(g.chord[1], 5), "ngv_throat_area_m2": float(f"{g.cascades()[0].A_throat:.4g}"),
                             "rotor_throat_area_m2": float(f"{g.cascades()[1].A_throat:.4g}")},
        "at_D10_120krpm": None if q is None else {"m4_model_kg_s": round(q.mass_flow_kg_s, 4), "m4_meas_kg_s": float(TOP.m4_kg_s),
                                                    "eta_tt_model": round(q.eta_tt, 4), "eta_tt_murray": float(TOP.turb_eta),
                                                    "ngv_exit_mach": round(q.stator_exit_mach, 3), "choked": q.choked,
                                                    "exit_swirl_deg": round(q.exit_flow_angle_deg, 1)},
        "per_speed_line": per_line(d),
        "exit_velocity_V5": {"calibration": {"err_pct": stats(v[v.set == "calibration"].err_pct),
                                             "within_published_uncertainty": f"{int(v[v.set == 'calibration'].within_pub.sum())}/{int((v.set == 'calibration').sum())}"},
                             "held_out": {"err_pct": stats(v[v.set == "held_out"].err_pct),
                                          "within_published_uncertainty": f"{int(v[v.set == 'held_out'].within_pub.sum())}/{int((v.set == 'held_out').sum())}"}},
        "held_out_all": {"Mcorr_err_pct": stats(d[(d.set == "held_out") & (d.figure == "5.16")].err),
                         "eta_err_pts": stats(d[(d.set == "held_out") & (d.figure == "5.18")].err)},
        "eta_all_lines_err_pts": stats(d[d.figure == "5.18"].err),
        "low_speed_le_44krpm_all_points": {"Mcorr_err_pct": stats(low[low.figure == "5.16"].err),
                                            "eta_err_pts": stats(low[low.figure == "5.18"].err)},
    }  # fmt: skip
    return out, d, v


def main() -> int:
    d0 = predict(X0)
    f_F = run_fit(("5.16", "V5"))
    f_FE = run_fit(("5.16", "5.18"))
    f_Q = run_fit(("5.16",))
    profile = {}
    for lam in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        fr = run_fit(("5.16", "V5"), x0=(lam, f_F["p"][1], f_F["p"][2]), fixed={0: lam})
        profile[lam] = {
            "rss": round(fr["rss"], 5),
            "ngv_gauging_deg": round(fr["p"][1], 2),
            "rotor_gauging_deg": round(fr["p"][2], 2),
        }
    rss_min = min(v["rss"] for v in profile.values())
    s2 = f_F["s2"]
    for (
        v
    ) in profile.values():  # profile-likelihood interval: delta RSS <= s^2 (1 sigma, 1 parameter)
        v["delta_rss_over_s2"] = round((v["rss"] - min(rss_min, f_F["rss"])) / s2, 2)
    sF, dF, vF = summarize(
        f_F, "F: flow (Fig 5.16) + exit velocity (App D V5); efficiency held out"
    )
    sFE, dFE, _ = summarize(f_FE, "FE: flow (Fig 5.16) + efficiency (Fig 5.18)")
    sQ, _dQ, _ = summarize(f_Q, "Q: flow (Fig 5.16) only")
    res = {
        "status": "CALIBRATED, NOT MEASURED (hub/tip and blade angles); counts, tip diameter and clearance published (Xu et al. 2021, a KJ66 variant)",
        "data": "Murray 2009 Figs 5.16/5.18 digitized [MEAS] (efficiency DERIVED by Murray); App D V5 (CALC_MEAS)",
        "split": {"calibration_lines_krpm": list(CAL_LINES), "held_out_lines_krpm": list(HELD_OUT),
                  "appD_calibration": sorted(D_CAL), "appD_held_out": sorted(set(AD.figure) - D_CAL)},
        "adopted": sF,
        "alternative_flow_plus_efficiency": sFE,
        "alternative_flow_only": sQ,
        "hub_tip_profile_fit_F": {float(k): v for k, v in profile.items()},
        "initial_guess": {"p": list(X0), "per_speed_line": per_line(d0)},
    }  # fmt: skip
    dF.to_csv(OUT / "turbine_calibration_murray.csv", index=False)
    vF.to_csv(OUT / "turbine_calibration_murray_v5.csv", index=False)
    plot(dF, dFE, d0)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "turbine_calibration_murray.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

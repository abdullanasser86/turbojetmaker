"""Validation against Murray's measured KJ-66 maps and operating lines (Chapter 5, digitized; user request 2026-10-25).

Data: validation/cases/murray2009/murray2009_ch5_maps_digitized.csv (Figs 5.12/5.14/5.16/5.18) and
murray2009_ch5_operating_lines_digitized.csv (Figs 5.1-5.3). Murray interpolated the map points to constant corrected
speed (p. 87), so each point's speed class is its corrected speed.

1. Compressor (KJ-66 model with the STORED calibration -- no refit): standard-day inlet (288.15 K, 101.325 kPa), so
   model mass flow and speed are the corrected values. Per speed line:
     PR error at the measured corrected mass flow; corrected-mass-flow error = model flow at the measured PR
     (on the speed line's stable, falling branch) minus measured; efficiency error at the Fig 5.14 points.
   Split: the stored calibration was fitted on measured 90 krpm data -> the 90 krpm line is "near-calibration",
   all other lines are VALIDATION. Contradiction test (reported, not acted on): a validation line whose PR errors all
   share one sign and exceed the published uncertainty.
2. Turbine (INDICATIVE, KJ-66 turbine geometry unknown): replica stages psi 1.0/1.5/2.0 (phi 0.7, R 0.5) sized at the
   120 krpm Appendix D point; each map point evaluated at its corrected speed (T04 = 800 K, P04 = 150 kPa; corrected
   quantities are independent of that choice) and at the measured total-to-total PR; corrected air mass flow and
   efficiency compared.
3. Engine operating lines (turbojet nozzle runs): the round-3 replicas (validation/phase4/murray2009_engine.csv, TIT
   fitted at 120 krpm only) vs Fig 5.1 (PR), 5.2 (corrected flow) and 5.3 (TIT, calculated by Murray) at corrected speed.

Usage:  uv run python scripts/phase4/murray_maps_validation.py
Output: validation/phase4/murray2009_maps_validation.yaml, murray2009_maps_*.csv, murray2009_maps.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from turboml.gas import thermo
from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry
from turboml.physics.turbine_design import design_stage

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "validation" / "cases" / "murray2009"
OUT = ROOT / "validation" / "phase4"
MAPS = pd.read_csv(CASE / "murray2009_ch5_maps_digitized.csv", comment="#", dtype={"figure": str})
LINES = pd.read_csv(
    CASE / "murray2009_ch5_operating_lines_digitized.csv", comment="#", dtype={"figure": str}
)
AD = pd.read_csv(CASE / "murray2009_appD_running_data.csv", comment="#")
CAL = yaml.safe_load(
    (ROOT / "configs" / "components" / "kj66_compressor_calibration.yaml").read_text(
        encoding="utf-8"
    )
)
TREF, PREF = 288.15, 101325.0
CAL_SPEED = 90


def stats(e) -> dict:
    e = pd.Series(e).dropna()
    if e.empty:
        return {"n": 0}
    return {"mean": round(float(e.mean()), 3), "rms": round(float((e**2).mean() ** 0.5), 3),
            "max_abs": round(float(e.abs().max()), 3), "n": int(e.size)}  # fmt: skip


def compressor() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    geom = CentrifugalCompressorGeometry.from_yaml()
    c = CentrifugalCompressor(geom, delta_sigma=CAL["delta_sigma"]["value"], diffuser_cf=CAL["diffuser_cf"]["value"],
                              T0_in_K=TREF, p0_in_Pa=PREF)  # fmt: skip
    flows = np.linspace(0.03, 0.30, 109)
    lines = {}
    for n in sorted(MAPS[MAPS.component == "compressor"].speed_class_krpm.unique()):
        sl = c.speed_line(n * 1e3, flows)
        lines[n] = sl[sl.converged][["mass_flow_kg_s", "pr_tt", "eta_tt"]].reset_index(drop=True)
    pr_pts = MAPS[MAPS.figure == "5.12"].copy()
    eta_pts = MAPS[MAPS.figure == "5.14"].copy()
    rows = []
    for r in pr_pts.itertuples():
        sl = lines[r.speed_class_krpm]
        p = c.evaluate(r.Mcorr_kg_s, r.speed_class_krpm * 1e3)
        # corrected-mass-flow offset: closest approach of the model speed line to the measured point, distances scaled by
        # the published uncertainties (well conditioned also where the speed line is flat, unlike inverting PR(m))
        u_m, u_pr = max(r.u_pub_Mcorr, r.u_dig_Mcorr_kg_s), max(r.u_pub_PR, r.u_dig_PR)
        mm = np.linspace(sl.mass_flow_kg_s.min(), sl.mass_flow_kg_s.max(), 2000)
        pp = np.interp(mm, sl.mass_flow_kg_s, sl.pr_tt)
        k = int(np.argmin(((mm - r.Mcorr_kg_s) / u_m) ** 2 + ((pp - r.PR) / u_pr) ** 2))
        m_at_pr = float(mm[k])
        rows.append({"speed_krpm": r.speed_class_krpm, "meas_Mcorr": r.Mcorr_kg_s, "meas_PR": r.PR,
                     "u_PR": max(r.u_pub_PR, r.u_dig_PR), "n_markers": r.n_markers_est,
                     "model_PR": p.pr_tt if p.converged else np.nan, "model_status": p.failure_reason or "ok",
                     "model_Mcorr_closest": m_at_pr, "model_range_kg_s": (round(float(sl.mass_flow_kg_s.min()), 3), round(float(sl.mass_flow_kg_s.max()), 3))})  # fmt: skip
    d = pd.DataFrame(rows)
    d["PR_err_pct"] = 100 * (d.model_PR / d.meas_PR - 1)
    d["Mcorr_err_pct"] = 100 * (d.model_Mcorr_closest / d.meas_Mcorr - 1)
    d["PR_within_unc"] = (d.model_PR - d.meas_PR).abs() <= d.u_PR
    erows = []
    for r in eta_pts.itertuples():
        p = c.evaluate(r.Mcorr_kg_s, r.speed_class_krpm * 1e3)
        erows.append({"speed_krpm": r.speed_class_krpm, "meas_Mcorr": r.Mcorr_kg_s, "meas_eta": r.eta,
                      "u_eta": max(r.u_pub_eta, r.u_dig_eta), "model_eta": p.eta_tt if p.converged else np.nan})  # fmt: skip
    e = pd.DataFrame(erows)
    e["eta_err_pts"] = 100 * (e.model_eta - e.meas_eta)
    e["eta_within_unc"] = (e.model_eta - e.meas_eta).abs() <= e.u_eta
    per = {}
    for n in sorted(d.speed_krpm.unique()):
        g, ge = d[d.speed_krpm == n], e[e.speed_krpm == n]
        sign = np.sign(g.PR_err_pct.dropna())
        per[int(n)] = {"set": "near_calibration" if n == CAL_SPEED else "validation",
                       "PR_err_pct": stats(g.PR_err_pct), "Mcorr_err_pct": stats(g.Mcorr_err_pct), "eta_err_pts": stats(ge.eta_err_pts),
                       "PR_within_published_uncertainty": f"{int(g.PR_within_unc.sum())}/{len(g)}",
                       "eta_within_published_uncertainty": f"{int(ge.eta_within_unc.sum())}/{len(ge)}",
                       "systematic_PR_bias_beyond_uncertainty": bool(len(sign) and abs(sign.sum()) == len(sign) and not g.PR_within_unc.any())}  # fmt: skip
    val_d, val_e = d[d.speed_krpm != CAL_SPEED], e[e.speed_krpm != CAL_SPEED]
    summ = {"per_speed_line": per,
            "validation_all_lines": {"PR_err_pct": stats(val_d.PR_err_pct), "Mcorr_err_pct": stats(val_d.Mcorr_err_pct),
                                     "eta_err_pts": stats(val_e.eta_err_pts),
                                     "PR_within_published_uncertainty": f"{int(val_d.PR_within_unc.sum())}/{len(val_d)}",
                                     "eta_within_published_uncertainty": f"{int(val_e.eta_within_unc.sum())}/{len(val_e)}"},
            "lines_with_systematic_PR_bias": [int(k) for k, v in per.items() if v["systematic_PR_bias_beyond_uncertainty"]]}  # fmt: skip
    d.attrs["lines"] = lines
    return d, e, summ


def turbine() -> tuple[pd.DataFrame, dict]:
    ref = AD[AD.figure == "D.10"].iloc[0]
    gas = thermo.products(ref.FAR)
    omega = ref.speed_krpm * 1e3 * math.pi / 30
    work = ref.turb_W_kW * 1e3 / ref.m4_kg_s
    T04, P04 = 800.0, 150e3
    m_pts = MAPS[MAPS.figure == "5.16"]
    e_pts = MAPS[MAPS.figure == "5.18"]
    rows, summ = [], {}
    for psi in (1.0, 1.5, 2.0):
        sz, t, _ = design_stage(
            ref.m4_kg_s, ref.T04_K, ref.P04_kPa * 1e3, omega, work, 0.7, psi, 0.5, gas=gas
        )
        t.set_gas(thermo.products(0.0155))

        def at(ncorr, pr_tt, t=t):
            om = ncorr * 1e3 * math.sqrt(T04 / TREF) * math.pi / 30

            def f(pr_ts):
                q = t.evaluate(T04, P04, om, pr_ts)
                return (P04 / q.p0_out_Pa - pr_tt) if q.converged else float("nan")

            try:
                pr_ts = brentq(f, pr_tt * 1.0005, pr_tt * 1.6, xtol=1e-6)
            except ValueError:
                return None
            return t.evaluate(T04, P04, om, pr_ts)

        for fig, pts, key in (("5.16", m_pts, "Mcorr_kg_s"), ("5.18", e_pts, "eta")):
            for r in pts.itertuples():
                q = at(r.speed_class_krpm, r.PR)
                row = {"psi": psi, "figure": fig, "speed_krpm": r.speed_class_krpm, "meas_PR": r.PR, "meas": getattr(r, key),
                       "u": max(r.u_pub_Mcorr if fig == "5.16" else r.u_pub_eta, getattr(r, f"u_dig_{key}"))}  # fmt: skip
                if q is not None and q.converged:
                    m_air = (
                        q.mass_flow_kg_s / 1.0155
                    )  # Murray's turbine corrected flow uses the AIR mass flow
                    row["model"] = (
                        m_air * math.sqrt(T04 / TREF) * PREF / P04 if fig == "5.16" else q.eta_tt
                    )
                rows.append(row)
    d = pd.DataFrame(rows)
    d["err"] = np.where(d.figure == "5.16", 100 * (d.model / d.meas - 1), 100 * (d.model - d.meas))
    for psi, g in d.groupby("psi"):
        summ[f"psi_{psi}"] = {
            "corrected_mass_flow_err_pct": stats(g[g.figure == "5.16"].err),
            "eta_err_pts": stats(g[g.figure == "5.18"].err),
            "per_speed_line": {int(n): {"Mcorr_err_pct": stats(gg[gg.figure == "5.16"].err), "eta_err_pts": stats(gg[gg.figure == "5.18"].err)}
                               for n, gg in g.groupby("speed_krpm")},
        }  # fmt: skip
    summ["note"] = ("INDICATIVE: assumed replica stages (geometry unknown); Murray's turbine efficiency is derived from the "
                    "compressor work (published uncertainty +-0.08-0.34)")  # fmt: skip
    return d, summ


def engine_lines() -> tuple[pd.DataFrame, dict]:
    eng = pd.read_csv(OUT / "murray2009_engine.csv")
    eng = eng[eng.model_converged]
    T_amb, p_amb = 294.7, 98.9
    eng = eng.assign(corr_speed=eng.speed_krpm * math.sqrt(TREF / T_amb),
                     model_Mcorr=eng.model_m_kg_s * math.sqrt(T_amb / TREF) * 101.325 / p_amb)  # fmt: skip
    lines = LINES
    env_pr = lines[lines.figure == "5.1"].sort_values("corrected_speed_krpm")
    env_m = lines[lines.figure == "5.2"].sort_values("corrected_speed_krpm")
    tit = lines[(lines.figure == "5.3") & lines.series.str.startswith("Nozzle")]
    rows = []
    lo_env, hi_env = env_pr.corrected_speed_krpm.min(), env_pr.corrected_speed_krpm.max()
    eng = eng[
        (eng.corr_speed >= lo_env - 0.5) & (eng.corr_speed <= hi_env + 0.5)
    ]  # envelope digitized 48-120 krpm only
    for r in eng.itertuples():
        run = r.run.split(",")[0]
        t = tit[tit.series == run].sort_values("corrected_speed_krpm")
        rows.append({"run": run, "psi": r.psi, "corr_speed": r.corr_speed,
                     "model_PR": r.model_PR, "env_PR_min": np.interp(r.corr_speed, env_pr.corrected_speed_krpm, env_pr.value_min),
                     "env_PR_max": np.interp(r.corr_speed, env_pr.corrected_speed_krpm, env_pr.value_max),
                     "model_Mcorr": r.model_Mcorr, "env_M_min": np.interp(r.corr_speed, env_m.corrected_speed_krpm, env_m.value_min),
                     "env_M_max": np.interp(r.corr_speed, env_m.corrected_speed_krpm, env_m.value_max),
                     "model_TIT": r.model_tit_K,
                     "meas_TIT": np.interp(r.corr_speed, t.corrected_speed_krpm, t.value_min) if len(t) and t.corrected_speed_krpm.min() <= r.corr_speed <= t.corrected_speed_krpm.max() else np.nan})  # fmt: skip
    d = pd.DataFrame(rows)

    def outside(v, lo, hi):
        return np.where(v < lo, v - lo, np.where(v > hi, v - hi, 0.0))

    d["PR_outside_envelope"] = outside(d.model_PR, d.env_PR_min, d.env_PR_max)
    d["Mcorr_outside_envelope_pct"] = (
        100 * outside(d.model_Mcorr, d.env_M_min, d.env_M_max) / d.env_M_min
    )
    d["TIT_err_K"] = d.model_TIT - d.meas_TIT
    summ = {}
    for psi, g in d.groupby("psi"):
        low = g[g.corr_speed < 60]
        summ[f"psi_{psi}"] = {"PR_outside_all_run_envelope": stats(g.PR_outside_envelope),
                              "Mcorr_outside_all_run_envelope_pct": stats(g.Mcorr_outside_envelope_pct),
                              "TIT_err_K_vs_same_nozzle_run": stats(g.TIT_err_K),
                              "TIT_err_K_below_60krpm_corrected": stats(low.TIT_err_K)}  # fmt: skip
    return d, summ


def plot(cd, ce, td, ed) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = plt.cm.viridis(np.linspace(0, 0.9, 9))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    for i, (n, sl) in enumerate(sorted(cd.attrs["lines"].items())):
        ax.plot(sl.mass_flow_kg_s, sl.pr_tt, color=cols[i], lw=1)
        g = cd[cd.speed_krpm == n]
        ax.errorbar(
            g.meas_Mcorr,
            g.meas_PR,
            yerr=g.u_PR,
            fmt="o",
            color=cols[i],
            ms=4,
            capsize=2,
            label=f"{n:g} krpm",
        )
    ax.set(xlabel="corrected mass flow [kg/s]", ylabel="PR (tt)", xlim=(0.03, 0.26), ylim=(1.0, 2.6),
           title="Compressor: model lines vs Murray Fig 5.12 (points)")  # fmt: skip
    ax.legend(fontsize=6, ncol=3, frameon=False)
    ax = axes[0, 1]
    for i, (n, sl) in enumerate(sorted(cd.attrs["lines"].items())):
        ax.plot(sl.mass_flow_kg_s, sl.eta_tt, color=cols[i], lw=1)
        g = ce[ce.speed_krpm == n]
        ax.errorbar(
            g.meas_Mcorr,
            g.meas_eta,
            yerr=g.u_eta,
            fmt="o",
            color=cols[i],
            ms=3,
            capsize=1.5,
            alpha=0.8,
        )
    ax.set(xlabel="corrected mass flow [kg/s]", ylabel="efficiency (tt)", xlim=(0.03, 0.26), ylim=(0.5, 0.9),
           title="Compressor efficiency vs Fig 5.14")  # fmt: skip
    ax = axes[1, 0]
    g = td[(td.figure == "5.16")]
    m = g[g.psi == 1.5]
    ax.errorbar(
        m.meas_PR, m.meas, yerr=m.u, fmt="o", color="k", ms=4, capsize=2, label="Murray Fig 5.16"
    )
    for psi, ls in ((1.0, ":"), (1.5, "-"), (2.0, "--")):
        gg = g[g.psi == psi].sort_values("meas_PR")
        ax.plot(gg.meas_PR, gg.model, ls=ls, color="#2a78d6", label=f"replica psi {psi}")
    ax.set(
        xlabel="turbine PR (tt)",
        ylabel="corrected air mass flow [kg/s]",
        title="Turbine (indicative): flow capacity",
    )
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 1]
    for run, gg in ed[ed.psi == 1.5].groupby("run"):
        gg = gg.sort_values("corr_speed")
        ax.plot(gg.corr_speed, gg.model_TIT, label=f"model, {run} replica")
    t = LINES[(LINES.figure == "5.3") & LINES.series.str.startswith("Nozzle")]
    for run, gg in t.groupby("series"):
        ax.plot(gg.corrected_speed_krpm, gg.value_min, "o--", ms=3, label=f"Murray Fig 5.3 {run}")
    ax.set(
        xlabel="corrected speed [krpm]",
        ylabel="TIT [K] (Murray: calculated)",
        title="Turbojet TIT vs speed",
    )
    ax.legend(fontsize=6, frameon=False)
    for a in axes.flat:
        a.grid(alpha=0.3)
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / "murray2009_maps.png", dpi=140)
    plt.close(fig)


def main() -> int:
    cd, ce, cs = compressor()
    td, ts = turbine()
    ed, es = engine_lines()
    cd.to_csv(OUT / "murray2009_maps_compressor_pr.csv", index=False)
    ce.to_csv(OUT / "murray2009_maps_compressor_eta.csv", index=False)
    td.to_csv(OUT / "murray2009_maps_turbine.csv", index=False)
    ed.to_csv(OUT / "murray2009_maps_engine_lines.csv", index=False)
    plot(cd, ce, td, ed)
    res = {"compressor": cs, "turbine_indicative": ts, "engine_operating_lines": es}
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "murray2009_maps_validation.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Second KJ-66 validation case: Murray (2009) running data (user request 2026-10-25, item 2).

Data (validation/cases/murray2009/):
  * murray2009_appD_running_data.csv -- Appendix D tables (TRANSCRIBED, MEAS / calculated-from-MEAS), exhaust-diffuser
    configuration, 48-120 krpm, June 30 2009;
  * murray2009_runplot_plateaus_digitized.csv -- Appendix C run plots (DIGITIZED [MEAS]), incl. the three NOZZLE
    (turbojet) runs down to 28 krpm.
The thesis' map figures 5.12-5.19 and Figs 5.1-5.3 were not in the uploaded extract (pp. 121-171 only).

1. Compressor (KJ-66 compressor model, calibration from configs/components/kj66_compressor_calibration.yaml): evaluated
   at each Appendix D point's measured speed, air mass flow and compressor-inlet state (T02, P02); compared with the
   measured PR = P03/P02 and Murray's efficiency (from T03/T02 and PR). Split: the stored calibration was fitted on the
   90 krpm measured line (as plotted by Xiang et al.); points within 3 krpm of 90 krpm (D.6 88.0, D.12 91.2) are
   reported as "near-calibration", all others as VALIDATION.
2. Turbine: the KJ-66 turbine geometry is not available, so the meanline turbine model cannot be validated directly.
   Indicative check: replica stages (phi 0.7, R 0.5, psi 1.0 / 1.5 / 2.0, an ASSUMED family) sized at the 120 krpm
   Appendix D point (m4, T04, P04, work) and evaluated at the other points' T04, P04, speed and total-to-static PR;
   compared with the measured mass flow (flow capacity) and Murray's derived efficiency.
3. Turbojet engine (nozzle runs): replica engines designed at the 120 krpm nozzle-run point (W from the bellmouth
   head, PR from P03, Murray's ambient) with ONE free parameter, TIT (equivalently the nozzle area), set so the
   model's thermocouple-equivalent EGT equals the measured T5 reading at 120 krpm (calibration). The operating line
   at every other measured speed (28-112 krpm) is a prediction (validation): compressor delivery pressure, mass flow
   and EGT.

Usage:  uv run python scripts/phase4/murray_validation.py
Output: validation/phase4/murray2009_validation.yaml + murray2009_compressor.csv, _turbine.csv, _engine.csv
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
from turboml.physics.atmosphere import isa
from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry
from turboml.physics.engine import Engine, EngineFailure, design_engine, load_engine_config
from turboml.physics.turbine_design import design_stage

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "validation" / "cases" / "murray2009"
OUT = ROOT / "validation" / "phase4"
AD = pd.read_csv(CASE / "murray2009_appD_running_data.csv", comment="#")
RP = pd.read_csv(CASE / "murray2009_runplot_plateaus_digitized.csv", comment="#")
CAL = yaml.safe_load(
    (ROOT / "configs" / "components" / "kj66_compressor_calibration.yaml").read_text(
        encoding="utf-8"
    )
)
RECOVERY = 0.68  # Murray 2009, bare-wire probe normal to the flow (ASSUM)
ETA_M = 0.995  # Murray's mechanical efficiency (ASSUM) used in his T04 energy balance
NEAR_CAL = {"D.6", "D.12"}
P_AMB_KPA = 98.9  # Murray June 30 2009 (Appendix D P01); run 1 (June 25) ambient not tabulated -> same value, noted
T_AMB_K = 294.7
BELLMOUTH_C_M2 = float(
    np.mean((AD.m_air_kg_s / np.sqrt(1.168 * (AD.P01_kPa - AD.Ps1_kPa) * 1e3))[AD.speed_krpm > 95])
)


def stats(e: pd.Series) -> dict:
    return {"mean": round(float(e.mean()), 4), "rms": round(float((e**2).mean() ** 0.5), 4),
            "max_abs": round(float(e.abs().max()), 4), "n": int(e.size)}  # fmt: skip


def compressor_validation() -> tuple[pd.DataFrame, dict]:
    geom = CentrifugalCompressorGeometry.from_yaml()
    rows = []
    for r in AD.itertuples():
        c = CentrifugalCompressor(geom, delta_sigma=CAL["delta_sigma"]["value"], diffuser_cf=CAL["diffuser_cf"]["value"],
                                  T0_in_K=r.T02_K, p0_in_Pa=r.P02_kPa * 1e3)  # fmt: skip
        p = c.evaluate(r.m_air_kg_s, r.speed_krpm * 1e3)
        rows.append({"figure": r.figure, "speed_krpm": r.speed_krpm, "m_kg_s": r.m_air_kg_s, "set": "near_calibration" if r.figure in NEAR_CAL else "validation",
                     "meas_PR": r.comp_PR, "u_meas_PR": r.u_comp_PR, "meas_eta": r.comp_eta, "u_meas_eta": r.u_comp_eta,
                     "model_PR": p.pr_tt if p.converged else np.nan, "model_eta": p.eta_tt if p.converged else np.nan,
                     "model_status": p.failure_reason or "ok"})  # fmt: skip
    d = pd.DataFrame(rows)
    d["PR_rel_err_pct"] = 100 * (d.model_PR / d.meas_PR - 1)
    d["eta_err_pts"] = 100 * (d.model_eta - d.meas_eta)
    d["PR_within_meas_unc"] = (d.model_PR - d.meas_PR).abs() <= d.u_meas_PR
    d["eta_within_meas_unc"] = (d.model_eta - d.meas_eta).abs() <= d.u_meas_eta
    out = {}
    for s, g in d.groupby("set"):
        g = g.dropna(subset=["model_PR"])
        out[s] = {"PR_rel_err_pct": stats(g.PR_rel_err_pct), "eta_err_pts": stats(g.eta_err_pts),
                  "PR_within_meas_uncertainty": f"{int(g.PR_within_meas_unc.sum())}/{len(g)}",
                  "eta_within_meas_uncertainty": f"{int(g.eta_within_meas_unc.sum())}/{len(g)}"}  # fmt: skip
    low = d[(d.set == "validation") & (d.speed_krpm < 70)].dropna(subset=["model_PR"])
    out["validation_below_70krpm_(below_the_original_calibration_data)"] = {
        "PR_rel_err_pct": stats(low.PR_rel_err_pct),
        "eta_err_pts": stats(low.eta_err_pts),
    }
    out["model_failures"] = d.loc[d.model_status != "ok", ["figure", "model_status"]].to_dict(
        "records"
    )
    return d, out


def turbine_check() -> tuple[pd.DataFrame, dict]:
    ref = AD[AD.figure == "D.10"].iloc[0]
    rows, summary = [], {}
    for psi in (1.0, 1.5, 2.0):
        far = ref.FAR
        gas = thermo.products(far)
        omega = ref.speed_krpm * 1e3 * math.pi / 30
        work = ref.turb_W_kW * 1e3 / ref.m4_kg_s
        try:
            sz, t, pt = design_stage(
                ref.m4_kg_s, ref.T04_K, ref.P04_kPa * 1e3, omega, work, 0.7, psi, 0.5, gas=gas
            )
        except ValueError as e:
            summary[f"psi_{psi}"] = f"sizing failed: {e}"
            continue
        for r in AD.itertuples():
            t.set_gas(thermo.products(r.FAR))
            p = t.evaluate(
                r.T04_K, r.P04_kPa * 1e3, r.speed_krpm * 1e3 * math.pi / 30, r.P04_kPa / r.Ps5_kPa
            )
            rows.append({"psi": psi, "figure": r.figure, "speed_krpm": r.speed_krpm, "meas_m4": r.m4_kg_s,
                         "u_meas_m": r.u_m_air_kg_s, "model_m4": p.mass_flow_kg_s if p.converged else np.nan,
                         "meas_eta_tt_derived": r.turb_eta, "u_meas_eta": r.u_turb_eta,
                         "model_eta_tt": p.eta_tt if p.converged else np.nan, "status": p.failure_reason or "ok"})  # fmt: skip
    d = pd.DataFrame(rows)
    d["m_rel_err_pct"] = 100 * (d.model_m4 / d.meas_m4 - 1)
    d["eta_err_pts"] = 100 * (d.model_eta_tt - d.meas_eta_tt_derived)
    for psi, g in d.groupby("psi"):
        g = g.dropna(subset=["model_m4"])
        summary[f"psi_{psi}"] = {"flow_capacity_rel_err_pct": stats(g.m_rel_err_pct), "eta_err_pts": stats(g.eta_err_pts),
                                 "converged": f"{len(g)}/{len(AD)}"}  # fmt: skip
    summary["note"] = (
        "INDICATIVE ONLY: assumed replica stages (KJ-66 turbine geometry unknown); measured turbine "
        "efficiency is derived by Murray from the compressor work (eta_m 0.995) and has +-0.07-0.37 uncertainty"
    )
    return d, summary


def tc_reading(p) -> float:
    return p.T05_static_K + RECOVERY * (p.T05_K - p.T05_static_K)


def engine_nozzle_runs(cfg) -> tuple[pd.DataFrame, dict]:
    noz = RP[RP.configuration == "nozzle"].copy()
    rho1 = P_AMB_KPA * 1e3 / (287.05 * T_AMB_K)
    noz["m_kg_s"] = BELLMOUTH_C_M2 * np.sqrt(rho1 * noz.dP_inlet_kPa * 1e3)
    noz["u_m_kg_s"] = noz.m_kg_s * 0.5 * noz.u_dP_inlet_kPa / noz.dP_inlet_kPa
    # compressor-inlet total pressure: Appendix D shows P01 - P02 = 0.16-0.60 kPa (inlet duct loss); interpolate in speed
    ads = AD.sort_values("speed_krpm")
    loss = np.interp(noz.speed_krpm, ads.speed_krpm, ads.P01_kPa - ads.P02_kPa)
    noz["PR_meas"] = (noz.P03_gauge_kPa + P_AMB_KPA) / (P_AMB_KPA - loss)
    # ambient as an ISA offset: p = 98.9 kPa -> altitude, T = 294.7 K -> dT_isa
    alt = brentq(lambda h: isa(h, 0.0).p_Pa - P_AMB_KPA * 1e3, -500, 2000)
    dT = T_AMB_K - isa(alt, 0.0).T_K
    rows, summary = [], {"ambient": {"p_kPa": P_AMB_KPA, "T_K": T_AMB_K, "isa_alt_m": round(alt, 1), "dT_isa_K": round(dT, 2)},
                         "bellmouth_C_m2": round(BELLMOUTH_C_M2, 6)}  # fmt: skip
    for run, g in noz.groupby("run"):
        top = g.loc[g.set_speed_krpm.idxmax()]
        for psi in (1.0, 1.5, 2.0):
            base = {"W_air_kg_s": float(top.m_kg_s), "pr_c": float(top.PR_meas), "n_rpm": float(top.speed_krpm) * 1e3,
                    "turb_phi": 0.7, "turb_psi": psi, "turb_reaction": 0.5, "alt_m": alt, "mach": 0.0, "dT_isa_K": dT}  # fmt: skip
            target = float(top.T5_probe_C) + 273.15

            def f(tit, base=base, target=target):
                try:
                    return (
                        tc_reading(design_engine({**base, "tit_K": tit}, cfg).design_point) - target
                    )
                except EngineFailure:
                    return float("nan")

            try:
                tit = brentq(f, 800.0, 1250.0, xtol=0.05)
                d = design_engine({**base, "tit_K": tit}, cfg)
            except (ValueError, EngineFailure) as e:
                summary[f"{run} psi {psi}"] = f"calibration failed: {e}"
                continue
            eng = Engine(d, cfg)
            for r in g.itertuples():
                p = eng.off_design(r.speed_krpm * 1e3, alt, 0.0, dT)
                row = {"run": run, "psi": psi, "set_speed_krpm": r.set_speed_krpm, "speed_krpm": r.speed_krpm,
                       "role": "calibration (TIT)" if r.Index == top.name else "validation",
                       "meas_T5_probe_K": r.T5_probe_C + 273.15, "u_meas_T5_K": r.u_T5_probe_C,
                       "meas_P03_gauge_kPa": r.P03_gauge_kPa, "meas_m_kg_s": r.m_kg_s, "u_meas_m": r.u_m_kg_s,
                       "meas_PR": r.PR_meas, "model_converged": p.converged, "model_status": p.failure_reason or "ok",
                       "design_tit_K": tit}  # fmt: skip
                if p.converged:
                    row.update(model_T5_tc_K=tc_reading(p), model_T05_total_K=p.T05_K, model_tit_K=p.tit_K,
                               model_P03_gauge_kPa=p.p03_Pa / 1e3 - P_AMB_KPA, model_m_kg_s=p.W_air_kg_s, model_PR=p.pr_c,
                               model_eta_c=p.eta_c, model_eta_t=p.eta_t_ts)  # fmt: skip
                rows.append(row)
    d = pd.DataFrame(rows)
    d["EGT_err_K"] = d.model_T5_tc_K - d.meas_T5_probe_K
    d["P03_err_kPa"] = d.model_P03_gauge_kPa - d.meas_P03_gauge_kPa
    d["m_rel_err_pct"] = 100 * (d.model_m_kg_s / d.meas_m_kg_s - 1)
    v = d[(d.role == "validation") & d.model_converged]
    for psi, g in v.groupby("psi"):
        summary[f"validation psi {psi}"] = {
            "EGT_err_K": stats(g.EGT_err_K), "P03_err_kPa": stats(g.P03_err_kPa), "m_rel_err_pct": stats(g.m_rel_err_pct),
            "EGT_within_digitizing_unc": f"{int((g.EGT_err_K.abs() <= g.u_meas_T5_K).sum())}/{len(g)}",
            "EGT_err_K_at_or_below_48krpm": stats(g[g.speed_krpm < 50].EGT_err_K),
            "converged": f"{len(g)}/{int((d[(d.role == 'validation') & (d.psi == psi)]).shape[0])}"}  # fmt: skip
    return d, summary


def measured_egt_trend() -> dict:
    """Measured EGT (thermocouple reading) vs speed, turbojet configuration (all nozzle runs pooled)."""
    noz = RP[RP.configuration == "nozzle"]
    out = {}
    for sp, g in noz.groupby("set_speed_krpm"):
        out[f"{sp:g} krpm"] = {"T5_probe_C_mean": round(float(g.T5_probe_C.mean()), 1),
                               "min": float(g.T5_probe_C.min()), "max": float(g.T5_probe_C.max()), "n": int(len(g))}  # fmt: skip
    ad = AD.sort_values("speed_krpm")
    out["exhaust_diffuser_config_T05_total_K (Appendix D)"] = {
        f"{r.speed_krpm:g}": r.T05_K for r in ad.itertuples()
    }
    return out


def plots(cd: pd.DataFrame, ed: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ink, muted = "#1f1f1e", "#6b6a63"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (run, g) in enumerate(ed[ed.psi == 1.5].groupby("run")):
        g = g.reset_index(drop=True)
        top = int(g.set_speed_krpm.idxmax())
        up, down = g.iloc[: top + 1], g.iloc[top + 1 :]
        ax.errorbar(up.speed_krpm, up.meas_T5_probe_K - 273.15, yerr=up.u_meas_T5_K, fmt="o", color=series[i], ms=5,
                    capsize=2, label=f"measured {run.split(',')[0]} (rising speed)")  # fmt: skip
        if down.empty:
            continue
        ax.errorbar(down.speed_krpm, down.meas_T5_probe_K - 273.15, yerr=down.u_meas_T5_K, fmt="o", mfc="white",
                    color=series[i], ms=5, capsize=2, label=f"measured {run.split(',')[0]} (falling speed)")  # fmt: skip
    m = ed[(ed.psi == 1.5) & ed.run.str.contains("Run 3") & ed.model_converged].sort_values(
        "speed_krpm"
    )
    ax.plot(
        m.speed_krpm,
        m.model_T5_tc_K - 273.15,
        color=ink,
        lw=2,
        label="model, run-3 replica (psi 1.5), TC reading",
    )
    for psi, ls in ((1.0, ":"), (2.0, "--")):
        mm = ed[(ed.psi == psi) & ed.run.str.contains("Run 3") & ed.model_converged].sort_values(
            "speed_krpm"
        )
        ax.plot(
            mm.speed_krpm,
            mm.model_T5_tc_K - 273.15,
            color=muted,
            lw=1.2,
            ls=ls,
            label=f"model, psi {psi}",
        )
    ax.set(xlabel="shaft speed [krpm]", ylabel="EGT, thermocouple reading [C]",
           title="KJ-66 turbojet (Murray 2009 nozzle runs, digitized) vs model; model TIT fitted at 120 krpm only")  # fmt: skip
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "murray2009_egt_vs_speed.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (mcol, ucol, qcol, lab) in zip(axes, [("meas_PR", "u_meas_PR", "model_PR", "pressure ratio"),
                                                ("meas_eta", "u_meas_eta", "model_eta", "efficiency (tt)")], strict=True):  # fmt: skip
        for s_, c in (("validation", series[0]), ("near_calibration", series[1])):
            g = cd[cd.set == s_]
            ax.errorbar(
                g[mcol],
                g[qcol],
                xerr=g[ucol],
                fmt="o",
                color=c,
                ms=5,
                capsize=2,
                label=s_.replace("_", " "),
            )
        lo, hi = float(min(cd[mcol].min(), cd[qcol].min())), float(
            max(cd[mcol].max(), cd[qcol].max())
        )
        ax.plot([lo, hi], [lo, hi], color=muted, lw=1, ls=":")
        ax.set(xlabel=f"measured {lab} (Murray App. D)", ylabel=f"model {lab}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "murray2009_compressor.png", dpi=150)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_engine_config()
    res: dict = {"sources": {"appD": "validation/cases/murray2009/murray2009_appD_running_data.csv (transcribed)",
                             "runplots": "validation/cases/murray2009/murray2009_runplot_plateaus_digitized.csv (digitized)"}}  # fmt: skip
    cd, res["compressor"] = compressor_validation()
    cd.to_csv(OUT / "murray2009_compressor.csv", index=False)
    td, res["turbine_indicative"] = turbine_check()
    td.to_csv(OUT / "murray2009_turbine.csv", index=False)
    ed, res["engine_nozzle_runs"] = engine_nozzle_runs(cfg)
    ed.to_csv(OUT / "murray2009_engine.csv", index=False)
    res["measured_egt_trend"] = measured_egt_trend()
    plots(cd, ed)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "murray2009_validation.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

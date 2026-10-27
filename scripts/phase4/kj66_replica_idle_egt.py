"""Idle-EGT re-check with a KJ-66 replica built from the REAL compressor geometry (R1 refit calibration) and the
KJ66-variant turbine with its CALIBRATED (not measured) hub/tip and blade angles (user request 2026-10-27, item 2/3).

Difference from the round-3/4 replicas (scripts/phase4/murray_validation.py): there the compressor was re-sized by the
design-mode rules and the turbine was an assumed replica stage (psi 1.0/1.5/2.0) with TIT fitted at the top speed.
Here both turbomachines are fixed geometry:
  compressor  configs/components/kj66_compressor.yaml (measured/published geometry) + kj66_compressor_calibration.yaml (R1),
  turbine     turboml.physics.turbine_design.stage_from_counts with configs/components/kj66_turbine.yaml
              (66 mm, 18 NGV, 24 blades, clearance 1 % of height published [Xu et al. 2021, KJ66 variant];
              hub/tip and angles calibrated on Murray's turbine maps -- "calibrated, not measured"),
and ONE free parameter per nozzle run, the propelling-nozzle area A9 (not published), set so the model's
thermocouple-equivalent EGT (recovery factor 0.68) equals the measured T5 reading at the run's top speed
(calibration). Every other speed of the run, in particular the low-speed/idle end (28-48 krpm), is a prediction.
Engine loss settings, combustor and the off-design solver are the dataset's (configs/engine_baseline.yaml); the
seed design that supplies the design-point loss-scaling reference is sized at the run's top point.

Two turbine variants: F (adopted calibration) and Q (flow-only fit; matches Murray's flow map within +-4 %).
DIAGNOSTIC (not a model change): the same replica with the turbine efficiency lowered by the mean gap between the
calibrated turbine and Murray's (derived) turbine efficiency, A9 re-fitted -- shows how much of the EGT error the
turbine-efficiency gap explains.

Usage:  uv run python scripts/phase4/kj66_replica_idle_egt.py
Output: validation/phase4/kj66_replica_idle_egt.yaml/.csv/.png
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from turboml.physics.compressor import CentrifugalCompressorGeometry
from turboml.physics.engine import (
    Engine,
    EngineFailure,
    copy_design,
    design_engine,
    load_engine_config,
)
from turboml.physics.turbine_design import stage_from_counts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"
_spec = importlib.util.spec_from_file_location(
    "murray_validation", ROOT / "scripts" / "phase4" / "murray_validation.py"
)
MV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(
    MV
)  # reuse its data preparation (bellmouth flow, PR, ambient) and tc_reading
KJ = yaml.safe_load(
    (ROOT / "configs" / "components" / "kj66_turbine.yaml").read_text(encoding="utf-8")
)
TURB = yaml.safe_load((OUT / "turbine_calibration_murray.yaml").read_text(encoding="utf-8"))


def turbine_params(variant: str) -> tuple[float, float, float, float]:
    """(hub/tip, NGV gauging, rotor gauging, rotor LE) of a turbine variant:
    F = adopted calibration (configs/components/kj66_turbine.yaml: flow + exit velocity),
    Q = flow-only fit (turbine_calibration_murray.yaml `alternative_flow_only`; matches the flow map, not V5),
    J = joint geometry + k_loss fit (turbine_joint_refit.yaml `joint_fit`, D-068; rotor LE from the same rule).
    """
    if variant == "J":
        import importlib.util

        jp = yaml.safe_load((OUT / "turbine_joint_refit.yaml").read_text(encoding="utf-8"))["joint_fit"]["params"]
        sp = importlib.util.spec_from_file_location("tcal", ROOT / "scripts" / "phase4" / "turbine_calibration_murray.py")
        tc = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(tc)
        p = (jp["hub_tip"], jp["ngv_gauging_deg"], jp["rotor_gauging_deg"])
        _g, le = tc.build(p, k_loss=jp["k_loss"])
        return (*p, float(le))
    if variant == "F":
        for k in ("hub_tip", "ngv_gauging_deg", "rotor_gauging_deg", "rotor_le_deg"):
            if KJ[k]["value"] is None:
                raise SystemExit(
                    f"configs/components/kj66_turbine.yaml: {k} not calibrated yet (run turbine_calibration_murray.py)"
                )
        return tuple(
            KJ[k]["value"]
            for k in ("hub_tip", "ngv_gauging_deg", "rotor_gauging_deg", "rotor_le_deg")
        )
    q = TURB["alternative_flow_only"]
    c = q["calibrated"]
    return (c["hub_tip"]["value"], c["ngv_gauging_deg"]["value"], c["rotor_gauging_deg"]["value"],
            q["rotor_le_deg_zero_incidence_at_D10"])  # fmt: skip


def turbine_geometry(variant: str = "F"):
    ht, g_ngv, g_rot, le = turbine_params(variant)
    return stage_from_counts(KJ["rotor_tip_radius_m"]["value"], ht, KJ["n_ngv"]["value"], KJ["n_rotor"]["value"],
                             g_ngv, g_rot, le, KJ["tip_clearance_height_ratio"]["value"])  # fmt: skip


def nozzle_runs() -> tuple[pd.DataFrame, float, float]:
    from turboml.physics.atmosphere import isa

    noz = MV.RP[MV.RP.configuration == "nozzle"].copy()
    rho1 = MV.P_AMB_KPA * 1e3 / (287.05 * MV.T_AMB_K)
    noz["m_kg_s"] = MV.BELLMOUTH_C_M2 * np.sqrt(rho1 * noz.dP_inlet_kPa * 1e3)
    ads = MV.AD.sort_values("speed_krpm")
    loss = np.interp(noz.speed_krpm, ads.speed_krpm, ads.P01_kPa - ads.P02_kPa)
    noz["PR_meas"] = (noz.P03_gauge_kPa + MV.P_AMB_KPA) / (MV.P_AMB_KPA - loss)
    alt = brentq(lambda h: isa(h, 0.0).p_Pa - MV.P_AMB_KPA * 1e3, -500, 2000)
    return noz, alt, MV.T_AMB_K - isa(alt, 0.0).T_K


def seed_design(top, alt, dT, cfg):
    """Design-mode engine at the run's top point; only supplies the loss-scaling reference and a starting A9
    (its own compressor and turbine are replaced). A few seed TITs are tried if the sizing fails."""
    base = {"W_air_kg_s": float(top.m_kg_s), "pr_c": float(top.PR_meas), "n_rpm": float(top.speed_krpm) * 1e3,
            "turb_phi": 0.7, "turb_psi": 1.5, "turb_reaction": 0.5, "alt_m": alt, "mach": 0.0, "dT_isa_K": dT}  # fmt: skip
    err = None
    for tit in (1000.0, 950.0, 1050.0, 900.0, 1100.0):
        try:
            return copy_design(design_engine({**base, "tit_K": tit}, cfg))
        except EngineFailure as e:
            err = e
    raise err


def replica(top, alt, dT, cfg, diag=None, variant="F"):
    d = seed_design(top, alt, dT, cfg)
    d.compressor_geometry = CentrifugalCompressorGeometry.from_yaml()
    d.turbine_geometry = turbine_geometry(variant)
    a9_seed = d.A9_m2
    target = float(top.T5_probe_C) + 273.15

    def make(a9):
        d.A9_m2 = a9
        e = Engine(d, cfg)
        e.diag_eta_delta = diag
        return e

    def f(a9):
        p = make(a9).off_design(top.speed_krpm * 1e3, alt, 0.0, dT)
        return MV.tc_reading(p) - target if p.converged else float("nan")

    grid = a9_seed * np.linspace(0.5, 2.0, 16)
    vals = [f(a) for a in grid]
    a9 = None
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            a9 = brentq(f, grid[i], grid[i + 1], xtol=1e-8)
            break
    return (
        (None if a9 is None else make(a9)),
        a9,
        a9_seed,
        list(zip(grid.tolist(), vals, strict=True)),
    )


def run_all(
    diag=None, label="model", variant="F", k_loss: float = 1.0
) -> tuple[pd.DataFrame, dict]:
    cfg = load_engine_config()
    cfg["turbine_design"] = {**cfg["turbine_design"], "loss_multiplier": {"value": float(k_loss)}}
    noz, alt, dT = nozzle_runs()
    rows, summ = [], {}
    for run, g in noz.groupby("run"):
        top = g.loc[g.set_speed_krpm.idxmax()]
        try:
            eng, a9, a9_seed, scan = replica(top, alt, dT, cfg, diag, variant)
        except EngineFailure as e:
            summ[run] = f"replica failed: {e}"
            continue
        if eng is None:
            summ[run] = {
                "A9_fit": "no root",
                "scan_A9_m2_vs_EGT_err_K": [
                    (round(a, 6), None if not np.isfinite(v) else round(v, 1)) for a, v in scan
                ],
            }
            continue
        summ[run] = {
            "A9_fit_m2": round(a9, 6),
            "A9_seed_m2": round(a9_seed, 6),
            "top_speed_krpm": float(top.speed_krpm),
        }
        for r in g.itertuples():
            p = eng.off_design(r.speed_krpm * 1e3, alt, 0.0, dT)
            row = {"case": label, "run": run.split(",")[0], "speed_krpm": r.speed_krpm,
                   "role": "calibration (A9)" if r.Index == top.name else "validation",
                   "meas_T5_probe_K": r.T5_probe_C + 273.15, "u_meas_T5_K": r.u_T5_probe_C, "meas_P03_gauge_kPa": r.P03_gauge_kPa,
                   "meas_m_kg_s": r.m_kg_s, "meas_PR": r.PR_meas, "model_converged": p.converged, "model_status": p.failure_reason or "ok"}  # fmt: skip
            if p.converged:
                row.update(model_T5_tc_K=MV.tc_reading(p), model_tit_K=p.tit_K, model_P03_gauge_kPa=p.p03_Pa / 1e3 - MV.P_AMB_KPA,
                           model_m_kg_s=p.W_air_kg_s, model_PR=p.pr_c, model_eta_c=p.eta_c, model_eta_t_ts=p.eta_t_ts,
                           model_pr_t=p.pr_t_ts, model_turbine_choked=p.turbine_choked, model_Fn_N=p.Fn_N,
                           model_TSFC_g_kN_s=1e6 * p.TSFC_kg_N_s)  # fmt: skip
            rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return d, summ
    d["EGT_err_K"] = d.model_T5_tc_K - d.meas_T5_probe_K
    d["P03_err_kPa"] = d.model_P03_gauge_kPa - d.meas_P03_gauge_kPa
    d["m_err_pct"] = 100 * (d.model_m_kg_s / d.meas_m_kg_s - 1)
    v = d[(d.role == "validation") & d.model_converged]
    summ["validation"] = {"EGT_err_K": MV.stats(v.EGT_err_K), "P03_err_kPa": MV.stats(v.P03_err_kPa), "m_err_pct": MV.stats(v.m_err_pct),
                          "EGT_within_digitizing_unc": f"{int((v.EGT_err_K.abs() <= v.u_meas_T5_K).sum())}/{len(v)}",
                          "converged": f"{int(d[d.role == 'validation'].model_converged.sum())}/{int((d.role == 'validation').sum())}",
                          "EGT_err_K_below_50krpm": MV.stats(v[v.speed_krpm < 50].EGT_err_K),
                          "EGT_err_K_28_32krpm": MV.stats(v[v.speed_krpm < 35].EGT_err_K),
                          "TIT_model_K_below_50krpm": MV.stats(v[v.speed_krpm < 50].model_tit_K)}  # fmt: skip
    return d, summ


def plot(d: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for (case, run), g in d[d.model_converged].groupby(["case", "run"]):
        g = g.sort_values("speed_krpm")
        ls = "-" if case.endswith("model") else ":"
        axes[0].plot(g.speed_krpm, g.model_T5_tc_K - 273.15, ls, label=f"{case}: {run}")
        axes[1].plot(g.speed_krpm, g.model_P03_gauge_kPa, ls, label=f"{case}: {run}")
        axes[2].plot(g.speed_krpm, g.model_m_kg_s, ls, label=f"{case}: {run}")
    meas = d[d.case == d.case.iloc[0]]
    for run, g in meas.groupby("run"):
        axes[0].errorbar(
            g.speed_krpm,
            g.meas_T5_probe_K - 273.15,
            yerr=g.u_meas_T5_K,
            fmt="o",
            ms=3,
            capsize=2,
            label=f"Murray {run}",
        )
        axes[1].plot(g.speed_krpm, g.meas_P03_gauge_kPa, "o", ms=3)
        axes[2].plot(g.speed_krpm, g.meas_m_kg_s, "o", ms=3)
    axes[0].set(
        xlabel="speed [krpm]",
        ylabel="EGT, thermocouple reading [C]",
        title="EGT (A9 fitted at the top speed)",
    )
    axes[1].set(
        xlabel="speed [krpm]", ylabel="P03 gauge [kPa]", title="Compressor delivery pressure"
    )
    axes[2].set(xlabel="speed [krpm]", ylabel="air mass flow [kg/s]", title="Air mass flow")
    axes[0].legend(fontsize=6, frameon=False)
    for a in axes:
        a.grid(alpha=0.3)
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / "kj66_replica_idle_egt.png", dpi=140)
    plt.close(fig)


def main() -> int:
    res = {
        "compressor_calibration": "R1 refit (configs/components/kj66_compressor_calibration.yaml)"
    }
    frames = []
    for variant, what in (
        ("F", "adopted turbine calibration (flow + exit velocity)"),
        ("Q", "flow-only turbine fit"),
    ):
        gap = (
            TURB["adopted" if variant == "F" else "alternative_flow_only"]["eta_all_lines_err_pts"][
                "mean"
            ]
            / 100.0
        )
        d, s = run_all(label=f"{variant}: model", variant=variant)
        d2, s2 = run_all(
            diag={"turbine": -gap},
            label=f"{variant}: diagnostic eta_t - {100 * gap:.1f} pts",
            variant=variant,
        )
        res[f"turbine_{variant}"] = {"what": what, "params_hub_tip_ngv_rotor_le": [float(v) for v in turbine_params(variant)],
                                     "model": s, "diagnostic_turbine_efficiency_gap_removed": {"eta_shift_pts": round(-100 * gap, 2), **s2}}  # fmt: skip
        frames += [d, d2]
    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv(OUT / "kj66_replica_idle_egt.csv", index=False)
    if not allr.empty:
        plot(allr)
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "kj66_replica_idle_egt.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

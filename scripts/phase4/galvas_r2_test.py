"""R2 test: Galvas (1973) NASA TN D-7487 against Murray's measured KJ-66 speed lines (user request 2026-10-27, round 6a).

What Galvas contains (primary source; page numbers are the report's printed page numbers):
* p. 2: "Shock losses in the rotor are neglected." p. 10 (concluding remarks): "For inducers with relative Mach numbers
  appreciably in excess of unity the predicted compressor total pressure ratio and efficiency are greater than that
  attainable experimentally. This is a result of neglecting shock losses in the inducer inlet."
  -> Galvas has NO loss term that grows with inducer relative Mach number.
* p. 5 incidence loss  dh_INC = W_L^2 / (2 c_p),  W_L = W_1MF sin|beta_opt - beta_1MF|  (rms streamline; optimum
  incidence from his ref. 4, incompressible). Grows with W1^2 only through the incidence mismatch; our loss set already
  has an incidence loss of this family (oh_aungier).
* p. 7 "Calculation of compressor choking flow": inducer choke when the rms incidence falls to the choking incidence
  of his ref. 9 ("data on inducer choking incidence levels as functions of relative Mach number and blade angle at the
  rms diameter for inducer pitch-chord ratios of 0.5 and thickness-chord ratios of 0.05"). Program listing p. 26 (DATA)
  and p. 29 (logic: IF (INC .LE. CINC) K=1, INC = beta_1MF - beta_1MFb in degrees, linear interpolation in M_W,rms
  and in blade angle between the 40/50/60 deg tables). This is a FLOW LIMIT (choke criterion), not a loss.
  Validity: rms blade angle 40-60 deg, M_W,rms 0.5-1.2, pitch/chord 0.5, t/c 0.05.
* p. 7 vaned diffuser recovery C_p** tabulated vs throat Mach 0.2-0.8 and blockage (listing p. 26, PREC1-5): it
  INCREASES slightly with Mach (e.g. 0.782 -> 0.802 at 2 % blockage, PREC3) and is "valid only in the subsonic flow
  regime" (p. 10). Our KJ-66 model has no vaned diffuser geometry (vane data unpublished), so it cannot be applied.

Test here: the inducer choke criterion on every measured Murray point (and on the phase4_v5 design points), with the
KJ-66 rms blade angle from the digitized midspan angle 42 deg (Ling 2007) scaled as tan(beta) ~ r (constant-lead
inducer, assumption) -> 45.9 deg; 42 deg reported as sensitivity.

Usage:  uv run python scripts/phase4/galvas_r2_test.py
Output: validation/phase4/galvas_r2_test.yaml, galvas_r2_test_points.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from turboml.gas import thermo
from turboml.physics.compressor import CentrifugalCompressor, CentrifugalCompressorGeometry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"
MAPS = pd.read_csv(
    ROOT / "validation/cases/murray2009/murray2009_ch5_maps_digitized.csv",
    comment="#",
    dtype={"figure": str},
)
CAL = yaml.safe_load(
    (ROOT / "configs/components/kj66_compressor_calibration.yaml").read_text(encoding="utf-8")
)
# Galvas 1973, listing p. 26 (DATA statements), transcribed
MW_RMS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
CINC = {40.0: np.array([-20.0, -12.0, -5.0, 1.5, 5.0, 8.0, 11.5, 14.5]),
        50.0: np.array([-15.0, -7.5, -3.0, 1.5, 4.0, 6.5, 9.0, 11.0]),
        60.0: np.array([-10.5, -5.0, -2.0, 1.5, 2.5, 4.0, 5.0, 5.0])}  # fmt: skip


def choking_incidence_deg(m_rms: float, beta_b_deg: float) -> float:
    """Galvas choking incidence (listing p. 29): interpolation in M_W,rms, then linear in blade angle between tables."""
    if not (MW_RMS[0] <= m_rms <= MW_RMS[-1]):
        return float("nan")  # outside the tabulated Mach range (the program has no rule there)
    lo, hi = (40.0, 50.0) if beta_b_deg <= 50.0 else (50.0, 60.0)
    c_lo, c_hi = np.interp(m_rms, MW_RMS, CINC[lo]), np.interp(m_rms, MW_RMS, CINC[hi])
    return float(c_lo + (beta_b_deg - lo) / 10.0 * (c_hi - c_lo))


def rms_state(c1_m_s: float, T01: float, omega: float, r_rms: float) -> tuple[float, float]:
    """(M_W,rms, beta_1MF [deg from meridional]) for axial inflow without pre-swirl."""
    air = thermo.air()
    u = omega * r_rms
    T1 = air.T_from_h(air.h(T01) - 0.5 * c1_m_s**2, T01)
    a1 = math.sqrt(air.gamma(T1) * air.R * T1)
    return math.hypot(c1_m_s, u) / a1, math.degrees(math.atan2(u, c1_m_s))


def main() -> int:
    g = CentrifugalCompressorGeometry.from_yaml()
    r_h, r_t = g.radius_hub_in_m, g.radius_tip_in_m
    r_rms, r_mid = math.sqrt(0.5 * (r_h**2 + r_t**2)), 0.5 * (r_h + r_t)
    beta_mid = 42.0
    beta_rms = math.degrees(math.atan(math.tan(math.radians(beta_mid)) * r_rms / r_mid))
    c = CentrifugalCompressor(g, delta_sigma=CAL["delta_sigma"]["value"], diffuser_cf=CAL["diffuser_cf"]["value"],
                              T0_in_K=288.15, p0_in_Pa=101325.0)  # fmt: skip
    rows = []
    for r in MAPS[MAPS.figure == "5.12"].itertuples():
        n = r.speed_class_krpm * 1e3
        p = c.evaluate(r.Mcorr_kg_s, n)
        if not p.converged:
            continue
        m_rms, b_flow = rms_state(p.inducer_axial_velocity_m_s, 288.15, n * math.pi / 30, r_rms)
        row = {"speed_krpm": r.speed_class_krpm, "Mcorr_kg_s": r.Mcorr_kg_s, "meas_PR": r.PR, "M_W_rms": m_rms,
               "M_W_tip_model": p.inlet_rel_mach_tip, "beta_1MF_deg": b_flow}  # fmt: skip
        for lab, bb in (("rms45.9", beta_rms), ("mid42", beta_mid)):
            inc = b_flow - bb
            ci = choking_incidence_deg(m_rms, bb)
            row[f"inc_{lab}_deg"] = inc
            row[f"choke_inc_{lab}_deg"] = ci
            row[f"galvas_choked_{lab}"] = bool(np.isfinite(ci) and inc <= ci)
        rows.append(row)
    d = pd.DataFrame(rows)
    per = {}
    for n, gg in d.groupby("speed_krpm"):
        per[int(n)] = {"M_W_rms": [round(float(gg.M_W_rms.min()), 3), round(float(gg.M_W_rms.max()), 3)],
                       "M_W_tip": [round(float(gg.M_W_tip_model.min()), 3), round(float(gg.M_W_tip_model.max()), 3)],
                       "incidence_rms_deg": [round(float(gg["inc_rms45.9_deg"].min()), 2), round(float(gg["inc_rms45.9_deg"].max()), 2)],
                       "choking_incidence_rms_deg": [round(float(np.nanmin(gg["choke_inc_rms45.9_deg"])), 2) if gg["choke_inc_rms45.9_deg"].notna().any() else None],
                       "margin_to_choke_deg": round(float(np.nanmin(gg["inc_rms45.9_deg"] - gg["choke_inc_rms45.9_deg"])), 2) if gg["choke_inc_rms45.9_deg"].notna().any() else None,
                       "galvas_choked_points_rms45.9": int(gg["galvas_choked_rms45.9"].sum()),
                       "galvas_choked_points_mid42": int(gg["galvas_choked_mid42"].sum()), "n": len(gg)}  # fmt: skip
    res = {"source": "Galvas 1973, NASA TN D-7487 (fetched from NTRS, public domain)",
           "finding": "no loss term grows with inducer relative Mach (shock losses neglected, pp. 2 and 10); the only Mach-dependent element applicable to the KJ-66 is the inducer choke criterion (flow limit)",
           "kj66_inducer": {"r_rms_m": round(r_rms, 5), "beta_b_mid_deg": beta_mid, "beta_b_rms_deg_assumed": round(beta_rms, 2)},
           "murray_points_per_line": per}  # fmt: skip
    # v5 design points
    D = pd.read_csv(ROOT / "datasets/phase4_v5/designs.csv")
    P = pd.read_csv(ROOT / "datasets/phase4_v5/points.csv")
    dp = P[P.point == "design"].set_index("design_id")
    ok = D[D.status == "ok"].set_index("design_id")
    vrows = []
    for i, r in ok.iterrows():
        rh, rt = r.geom_r1_hub_m, r.geom_r1_tip_m
        rr = math.sqrt(0.5 * (rh**2 + rt**2))
        bm = (
            r.geom_inducer_le_angle_deg
        )  # the model's single inducer blade angle = KJ-66 midspan value (42 deg)
        b_rms = math.degrees(math.atan(math.tan(math.radians(abs(bm))) * rr / (0.5 * (rh + rt))))
        q = dp.loc[i]
        air = thermo.air()
        rho01 = q.p02_Pa / (air.R * q.T02_K)
        area = math.pi * (rt**2 - rh**2)
        c1 = q.W_air_kg_s / (rho01 * area)
        for _ in range(30):  # static density at the inducer
            T1 = q.T02_K - c1**2 / (2 * air.cp(q.T02_K))
            rho1 = rho01 * (T1 / q.T02_K) ** (1 / (air.gamma(T1) - 1))
            c1 = q.W_air_kg_s / (rho1 * area)
        m_rms, b_flow = rms_state(c1, q.T02_K, q.n_rpm * math.pi / 30, rr)
        ci = choking_incidence_deg(m_rms, b_rms)
        vrows.append({"design_id": i, "U2": r.in_impeller_tip_speed_m_s, "regime": r.tip_speed_regime, "M_W_rms": m_rms,
                      "beta_b_rms": b_rms, "inc": b_flow - b_rms, "choke_inc": ci,
                      "in_validity": bool(40 <= b_rms <= 60 and np.isfinite(ci)),
                      "galvas_choked": bool(np.isfinite(ci) and 40 <= b_rms <= 60 and (b_flow - b_rms) <= ci)})  # fmt: skip
    v = pd.DataFrame(vrows)
    res["phase4_v5_design_points"] = {
        reg: {"designed": len(g2), "M_W_rms_range": [round(float(g2.M_W_rms.min()), 3), round(float(g2.M_W_rms.max()), 3)],
              "beta_b_rms_range_deg": [round(float(g2.beta_b_rms.min()), 1), round(float(g2.beta_b_rms.max()), 1)],
              "inside_galvas_validity": int(g2.in_validity.sum()), "galvas_inducer_choked": int(g2.galvas_choked.sum())}
        for reg, g2 in v.groupby("regime")}  # fmt: skip
    d.to_csv(OUT / "galvas_r2_test_points.csv", index=False)
    res = json.loads(json.dumps(res, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    (OUT / "galvas_r2_test.yaml").write_text(yaml.safe_dump(res, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(res, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

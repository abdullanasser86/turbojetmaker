"""Phase 4 dataset generator: sample engine designs, evaluate them with the coupled physics model, record
everything (including every failure) in a resumable, append-only file.

How a design is evaluated
-------------------------
1. Design variables come from a scrambled Sobol sequence (even coverage of the design box; the box and the
   sources of its bounds are in configs/bounds.yaml). A seeded 20 % of the designs also sample the compressor
   calibration factors inside their inflated uncertainty band (configs/components/kj66_compressor_calibration.yaml).
2. Design mode (`turboml.physics.engine.design_engine`) sizes compressor, turbine and nozzle. If that fails,
   the design is recorded with its failure label and all four points are marked "design_failed".
3. Mechanical/creep checks at the design point (`turboml.physics.mechanical`): feasibility, selected material,
   active constraints, and the raw stresses/temperatures, so feasibility can be recomputed later with other
   settings (life, safety factor, taper factor, pattern factor).
4. Off-design points (speed-controlled): M0.5 at 3 km at design speed, 80 % speed, idle (35 % speed),
   each with its own convergence flag and failure reason.

Every design is written as one JSON line as soon as it finishes (`designs.jsonl`). A restart skips the ids
already in the file, so an interrupted run (laptop sleep, crash) loses at most the designs in flight.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = ROOT / "configs" / "dataset_phase4.yaml"


@dataclass
class DatasetSpec:
    cfg: dict
    bounds: dict  # name -> (lower, upper, scale, key)
    calibration: dict

    @classmethod
    def load(cls, path: str | Path = DATASET_CONFIG) -> DatasetSpec:
        from turboml.config import load_config

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        b = load_config().bounds
        bounds = {}
        for name, v in cfg["design_variables"].items():
            bd = b[name]
            if not bd.is_set():
                raise ValueError(f"bound {name} is not set in configs/bounds.yaml")
            bounds[name] = (float(bd.lower), float(bd.upper), v["scale"], v["key"])
        cal = yaml.safe_load((ROOT / cfg["calibration_sampling"]).read_text(encoding="utf-8"))
        return cls(cfg, bounds, cal)


def sample_designs(spec: DatasetSpec) -> list[dict]:
    """Design-variable samples (scrambled Sobol, seeded) plus the per-design calibration factors."""
    cfg = spec.cfg
    names = list(spec.bounds)
    n = int(cfg["n_designs"])
    m = int(math.ceil(math.log2(n)))
    u = qmc.Sobol(d=len(names), scramble=True, seed=cfg["seed"]).random_base2(m)[:n]
    rng = np.random.default_rng(cfg["seed"] + 1)  # separate stream for the calibration sampling
    cal = spec.calibration
    frac = float(cal["phase4_sampling"]["fraction_of_designs"])
    ds_lo, ds_hi = cal["delta_sigma"][cal["phase4_sampling"]["range"]]
    cf_lo, cf_hi = cal["diffuser_cf"][cal["phase4_sampling"]["range"]]
    sampled = rng.random(n) < frac
    ds = np.where(sampled, rng.uniform(ds_lo, ds_hi, n), cal["delta_sigma"]["value"])
    cf = np.where(sampled, rng.uniform(cf_lo, cf_hi, n), cal["diffuser_cf"]["value"])
    out = []
    for i in range(n):
        x = {}
        for j, name in enumerate(names):
            lo, hi, scale, key = spec.bounds[name]
            if (
                scale == "piecewise"
            ):  # piecewise-linear map of the Sobol coordinate (e.g. 15 % of samples above a knot)
                knots = spec.cfg["design_variables"][name]["knots"]
                x[key] = float(
                    np.interp(
                        u[i, j],
                        [0.0, *[k[0] for k in knots], 1.0],
                        [lo, *[k[1] for k in knots], hi],
                    )
                )
            else:
                x[key] = lo * (hi / lo) ** u[i, j] if scale == "log" else lo + (hi - lo) * u[i, j]
        x.update({k: float(v) for k, v in spec.cfg["design_condition"].items()})
        out.append({"design_id": i, "inputs": x, "calibration_sampled": bool(sampled[i]),
                    "delta_sigma": float(ds[i]), "diffuser_cf": float(cf[i])})  # fmt: skip
    return out


def make_test_split(n: int, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask of the frozen held-out test designs (seeded permutation; independent of results)."""
    rng = np.random.default_rng(seed + 2)
    mask = np.zeros(n, bool)
    mask[rng.permutation(n)[: int(round(fraction * n))]] = True
    return mask


def _point_dict(p) -> dict:
    keys = ("converged", "failure_reason", "n_rpm", "alt_m", "mach", "W_air_kg_s", "far", "Wf_kg_s", "Fn_N", "Fg_N",
            "TSFC_kg_N_s", "pr_c", "eta_c", "P_comp_W", "tit_K", "pr_t_ts", "eta_t_ts", "P_turb_W", "turbine_choked",
            "T02_K", "p02_Pa", "T03_K", "p03_Pa", "p04_Pa", "T05_K", "p05_Pa", "V9_m_s", "nozzle_choked", "surge_flag",
            "compressor_tip_speed_m_s", "t0_rel_rotor_K", "iterations", "max_residual", "time_s")  # fmt: skip
    return {k: _plain(getattr(p, k)) for k in keys}


def _plain(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def evaluate_design(rec: dict, engine_cfg: dict, points: list[dict]) -> dict:
    """Evaluate one sampled design (never raises: every failure becomes a labelled record)."""
    from turboml.physics.engine import Engine, EngineFailure, design_engine

    t0 = time.perf_counter()
    out = {**rec, "status": "ok", "failure_reason": "", "points": {}}
    try:
        d = design_engine(
            rec["inputs"], engine_cfg, calibration=(rec["delta_sigma"], rec["diffuser_cf"])
        )
    except EngineFailure as e:
        out.update(status="design_failed", failure_reason=str(e))
    except Exception as e:  # noqa: BLE001 -- unexpected model error: recorded, not dropped
        out.update(status="design_failed", failure_reason=f"design_error: {type(e).__name__}: {e}"[:200],
                   traceback=traceback.format_exc()[-1500:])  # fmt: skip
    if out["status"] != "ok":
        for pt in points:
            out["points"][pt["name"]] = {"converged": False, "failure_reason": "design_failed"}
        out["time_s"] = time.perf_counter() - t0
        return out

    cg, tg, mech, sz = d.compressor_geometry, d.turbine_geometry, d.mechanical, d.turbine_sizing
    out["geometry"] = {
        "r2_m": cg.radius_out_m, "b2_m": cg.width_out_m, "r1_tip_m": cg.radius_tip_in_m, "r1_hub_m": cg.radius_hub_in_m,
        "diffuser_r_out_m": cg.diffuser_radius_out_m, "inducer_le_angle_deg": cg.leading_edge_angle_deg,
        "turb_r_mean_m": sz.r_mean_m, "turb_U_mean_m_s": sz.U_m_s, "turb_rotor_tip_r_m": tg.rotor_tip_radius_m,
        "turb_rotor_annulus_m2": tg.rotor_annulus_area_m2,
        "turb_hub_tip_rotor_out": tg.radius_hub_out[1] / tg.radius_tip_out[1],
        "turb_blades_stator": sz.n_blades[0], "turb_blades_rotor": sz.n_blades[1], "A9_m2": d.A9_m2,
        "turb_sizing_warnings": "; ".join(d.notes),
    }  # fmt: skip
    env = mech.checks["envelope"]
    creep = {k: {kk: _plain(vv) for kk, vv in v.items() if not isinstance(vv, list)} | {"active": v["active"]}
             for k, v in mech.checks["creep"].items()}  # fmt: skip
    out["mechanical"] = {"feasible": mech.feasible, "material": mech.material,
                         "premium_material_required": mech.premium_material_required,
                         "active_constraints": list(mech.infeasible_reasons), "tags": list(mech.tags),
                         "impeller_tip_speed_m_s": mech.checks["impeller_tip_speed"]["value_m_s"],
                         "an2_m2_rpm2": mech.checks["an2_m2_rpm2"], "creep": creep,
                         "diameter_m": env["diameter_m"], "length_m": env["length_m"], "dry_mass_kg": env["dry_mass_kg"]}  # fmt: skip
    out["mechanical"].update(
        {
            "material_status": mech.material_status,
            "turbine_design_range_flags": list(mech.checks["turbine_design_range_flags"]),
            "turbine_exit_mach": _plain(mech.checks["turbine_exit"]["mach"]),
            "turbine_exit_swirl_deg": _plain(mech.checks["turbine_exit"]["swirl_deg"]),
            "sensitivity": sensitivities(d),
        }
    )
    # variables derived by the sizing when the sampled set prescribes their counterparts (option C)
    out["derived"] = {"n_rpm": _plain(d.inputs["n_rpm"]), "turb_phi": _plain(d.inputs["turb_phi"]),
                      "turb_hub_tip": _plain(tg.radius_hub_out[1] / tg.radius_tip_out[1]),
                      "impeller_tip_speed_m_s": _plain(d.design_point.compressor_tip_speed_m_s),
                      "pr_c": _plain(d.design_point.pr_c), "comp_specific_speed_conventional": _plain(conventional_ns(d)),
                      "turb_hub_tip_max_mach": _plain(d.inputs.get("turb_hub_tip_max_mach", float("nan"))),
                      "turb_hub_tip_mach_binding": _plain(d.inputs.get("turb_hub_tip_mach_binding", float("nan")))}  # fmt: skip
    out["points"]["design"] = _point_dict(d.design_point)
    try:
        eng = Engine(d, engine_cfg)
        for pt in points:
            if pt["name"] == "design":
                continue
            p = eng.off_design(pt["n_frac"] * d.design_point.n_rpm, pt["alt_m"], pt["mach"])
            out["points"][pt["name"]] = _point_dict(p)
    except Exception as e:  # noqa: BLE001
        for pt in points:
            out["points"].setdefault(
                pt["name"],
                {
                    "converged": False,
                    "failure_reason": f"off_design_error: {type(e).__name__}: {e}"[:200],
                },
            )
    out["time_s"] = time.perf_counter() - t0
    return out


def conventional_ns(d) -> float:
    """Conventional compressor specific speed N_s = omega sqrt(Q01) / dh_is^0.75 (dimensionless) at the design point."""
    from turboml.gas import thermo

    air, p = thermo.air(), d.design_point
    T2s = air.T_isentropic(p.T02_K, p.pr_c)
    dh_is = air.h(T2s) - air.h(p.T02_K)
    q = p.W_air_kg_s * air.R * p.T02_K / p.p02_Pa
    return p.n_rpm * math.pi / 30 * math.sqrt(q) / dh_is**0.75


def sensitivities(d) -> dict:
    """Feasibility of one design under the sensitivity settings of configs/materials.yaml (user decision
    2026-10-25): blade taper At/Ah = 0.34 (K = 0.67) and pattern factor at the ends of its range. The default
    settings (At/Ah = 1, PF = 0.25) decide feasibility; these columns only show how much the decision depends
    on them."""
    from turboml.physics import mechanical

    cfg = yaml.safe_load(mechanical.MATERIALS.read_text(encoding="utf-8"))
    cases = {f"at_ah_{cfg['blade_tip_to_hub_area_ratio']['sensitivity']}": {
                 "tip_to_hub_area_ratio": float(cfg["blade_tip_to_hub_area_ratio"]["sensitivity"])}}  # fmt: skip
    for pf in cfg["combustor_pattern_factor"]["sensitivity"]:
        cases[f"pf_{pf}"] = {"pattern_factor": float(pf)}
    out = {}
    for name, ov in cases.items():
        r = mechanical.evaluate(**d.mech_inputs, mats=mechanical.MaterialSet.from_yaml(**ov))
        out[name] = {
            "feasible": r.feasible,
            "material": r.material,
            "active": list(r.infeasible_reasons),
        }
    return out


def evaluate_design_json(rec: dict, engine_cfg: dict, points: list[dict]) -> str:
    return json.dumps(evaluate_design(rec, engine_cfg, points), default=_plain)


def done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["design_id"])
            except json.JSONDecodeError:  # a line cut off by a crash: that design is re-run
                continue
    return ids

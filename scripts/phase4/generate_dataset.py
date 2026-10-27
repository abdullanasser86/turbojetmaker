"""Generate the Phase 4 engine dataset (resumable, parallel). See turboml/data/generator.py for what is computed.

Usage (laptop or container):
    uv run python scripts/phase4/generate_dataset.py                  # all designs in configs/dataset_phase4_v2.yaml (--config for v1)
    uv run python scripts/phase4/generate_dataset.py --limit 160      # pilot: the first 160 Sobol designs
    TURBOML_N_WORKERS=8 uv run python scripts/phase4/generate_dataset.py
Interrupt any time (Ctrl-C, sleep, crash) and run the same command again: finished designs are skipped.

Outputs (datasets/<name>/, git-ignored): designs.jsonl (one JSON record per design, append-only), designs.csv and
points.csv (flat tables, rebuilt at the end). Committed: data/splits/<name>_split.csv (frozen test split, written
once before any design is evaluated) and the run log in runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from turboml.compute import n_workers
from turboml.config import load_config
from turboml.data.generator import (
    DatasetSpec,
    done_ids,
    evaluate_design_json,
    make_test_split,
    sample_designs,
)
from turboml.physics.engine import load_engine_config
from turboml.runlog import start_run

ROOT = Path(__file__).resolve().parents[2]


def write_split(spec: DatasetSpec) -> Path:
    cfg = spec.cfg
    path = ROOT / "data" / "splits" / f"{cfg['name']}_split.csv"
    mask = make_test_split(int(cfg["n_designs"]), float(cfg["test_fraction"]), int(cfg["seed"]))
    df = pd.DataFrame(
        {"design_id": np.arange(len(mask)), "split": np.where(mask, "test", "train_val")}
    )
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(df):
            raise SystemExit(
                f"{path} exists and differs from the split implied by the config: refusing to change a frozen split"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def repair_tail(path: Path) -> None:
    """Drop a last line cut off by a crash (no trailing newline), so appends start on a fresh line."""
    if path.exists() and path.stat().st_size:
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            path.write_bytes(data[: data.rfind(b"\n") + 1])


def apply_engine_overrides(engine_cfg: dict, overrides: dict | None) -> dict:
    """Deep-merge a dataset config's `engine_overrides` (e.g. {turbine_design: {loss_multiplier: 2.1}}) into the
    engine config, so a dataset version can change a calibration without editing configs/engine_baseline.yaml.
    """
    if not overrides:
        return engine_cfg
    out = dict(engine_cfg)
    for k, v in overrides.items():
        out[k] = (
            apply_engine_overrides(dict(out.get(k, {})), v)
            if isinstance(v, dict) and "value" not in v
            else v
        )
    return out


def flatten(jsonl: Path, out_dir: Path, split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    split = pd.read_csv(split_csv).set_index("design_id").split
    drows, prows = [], []
    for r in recs:
        base = {"design_id": r["design_id"], "split": split[r["design_id"]], "status": r["status"],
                "failure_reason": r["failure_reason"], "calibration_sampled": r["calibration_sampled"],
                "delta_sigma": r["delta_sigma"], "diffuser_cf": r["diffuser_cf"], "time_s": r.get("time_s"),
                "tip_speed_regime": r.get("tip_speed_regime"), "compressor_band_scale": r.get("compressor_band_scale"),
                "turbine_loss_multiplier": r.get("turbine_loss_multiplier"),
                **{f"in_{k}": v for k, v in r["inputs"].items()}}  # fmt: skip
        g = r.get("geometry", {})
        m = r.get("mechanical", {})
        row = {
            **base,
            **{f"geom_{k}": v for k, v in g.items()},
            **{f"derived_{k}": v for k, v in r.get("derived", {}).items()},
        }
        if m:
            row.update({"feasible": m["feasible"], "material": m["material"], "premium_material_required": m["premium_material_required"],
                        "active_constraints": ";".join(m["active_constraints"]), "impeller_tip_speed_m_s": m["impeller_tip_speed_m_s"],
                        "an2_m2_rpm2": m["an2_m2_rpm2"], "diameter_m": m["diameter_m"], "length_m": m["length_m"],
                        "dry_mass_kg": m["dry_mass_kg"]})  # fmt: skip
            if "material_status" in m:
                row.update({"material_status": m["material_status"],
                            "turbine_design_range_flags": "; ".join(m["turbine_design_range_flags"]),
                            "turbine_exit_mach": m["turbine_exit_mach"], "turbine_exit_swirl_deg": m["turbine_exit_swirl_deg"]})  # fmt: skip
                for case, sv in m["sensitivity"].items():
                    row[f"sens_{case}_feasible"] = sv["feasible"]
                    row[f"sens_{case}_active"] = ";".join(sv["active"])
            for mat, c in m["creep"].items():
                for k in ("rotor_metal_T_K", "rotor_stress_centrifugal_Pa", "rotor_stress_gas_bending_Pa", "rotor_allowable_Pa",
                          "rotor_status", "ngv_stress_gas_bending_Pa", "ngv_allowable_T_K", "ngv_pf_max", "ngv_status", "feasible"):  # fmt: skip
                    row[f"creep_{mat}_{k}"] = c.get(k)
        drows.append(row)
        for name, p in r["points"].items():
            prows.append({"design_id": r["design_id"], "split": base["split"], "point": name, **p})
    designs, points = pd.DataFrame(drows).sort_values("design_id"), pd.DataFrame(prows).sort_values(
        ["design_id", "point"]
    )
    designs.to_csv(out_dir / "designs.csv", index=False)
    points.to_csv(out_dir / "points.csv", index=False)
    return designs, points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset_phase4_v2.yaml"))
    ap.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N designs (pilot)"
    )
    args = ap.parse_args()
    spec = DatasetSpec.load(args.config)
    cfg = spec.cfg
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    split_csv = write_split(spec)  # frozen BEFORE any evaluation
    jsonl = out_dir / "designs.jsonl"
    repair_tail(jsonl)
    designs = sample_designs(spec)
    todo_all = designs if args.limit is None else designs[: args.limit]
    done = done_ids(jsonl)
    todo = [d for d in todo_all if d["design_id"] not in done]
    workers = n_workers(load_config().compute.n_workers)
    run_dir = start_run(load_config(), f"dataset_{cfg['name']}")
    shutil.copy(args.config, run_dir / "dataset_config.yaml")
    print(
        f"{len(done)} done, {len(todo)} to run on {workers} workers; split {split_csv}; run log {run_dir}",
        flush=True,
    )
    engine_cfg = apply_engine_overrides(load_engine_config(), cfg.get("engine_overrides"))
    t0, k = time.time(), 0
    with jsonl.open("a", encoding="utf-8") as fh:
        for line in Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(evaluate_design_json)(d, engine_cfg, cfg["points"]) for d in todo
        ):
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            k += 1
            if k % 20 == 0 or k == len(todo):
                el = time.time() - t0
                print(
                    f"{k}/{len(todo)}  {el:.0f} s elapsed  ETA {el / k * (len(todo) - k):.0f} s",
                    flush=True,
                )
    d_df, p_df = flatten(jsonl, out_dir, split_csv)
    wall = time.time() - t0
    (run_dir / "timing.json").write_text(
        json.dumps({"designs_this_run": k, "wall_s": wall, "workers": workers}), encoding="utf-8"
    )
    print(
        f"wrote {out_dir / 'designs.csv'} ({len(d_df)} designs) and points.csv ({len(p_df)} rows); wall {wall:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

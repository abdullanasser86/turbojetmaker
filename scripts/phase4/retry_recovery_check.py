"""How many phase4_v5 pilot failures does the turbine-solver retry recover? (user request 2026-10-27, D-067)

Re-evaluates the first `--n` designs of phase4_v5 (the pilot) with the current code (restarts enabled), at the v5
settings (k_loss = 1) and at the v6 nominal k_loss (1.86), and compares with the stored v5 records point by point.
Nothing is written to datasets/.

Usage:  uv run python scripts/phase4/retry_recovery_check.py [--n 200]
Output: validation/phase4/retry_recovery_check.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from turboml.data.generator import DatasetSpec, evaluate_design, sample_designs
from turboml.physics.engine import load_engine_config

ROOT = Path(__file__).resolve().parents[2]
POINTS = ("design", "cruise_M05_3km", "part_speed_80", "idle_35")


def compare(old: dict, new: list) -> dict:
    trans, recovered, still, lost = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    retry_used, diffs = 0, []
    for r in new:
        o = old[r["design_id"]]
        trans[f"{o['status']} -> {r['status']}"] += 1
        for pn in POINTS:
            po, pq = o["points"].get(pn, {}), r["points"].get(pn, {})
            if pq.get("converged_on_retry"):
                retry_used += 1
            if not po.get("converged") and pq.get("converged"):
                recovered[f"{pn}: {po.get('failure_reason')}"] += 1
            elif po.get("converged") and not pq.get("converged"):
                lost[f"{pn}: {pq.get('failure_reason')}"] += 1
            elif not po.get("converged") and not pq.get("converged"):
                still[f"{pn}: {pq.get('failure_reason')}"] += 1
            elif po.get("converged") and pq.get("converged") and not pq.get("converged_on_retry"):
                diffs.append(abs(pq["Fn_N"] / po["Fn_N"] - 1) if po.get("Fn_N") else 0.0)
    return {"design_status": dict(trans), "points_recovered": dict(recovered), "points_newly_failed": dict(lost),
            "points_still_failed": dict(still), "points_converged_on_retry": retry_used,
            "unchanged_points_max_rel_Fn_change": float(np.max(diffs)) if diffs else None}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    spec = DatasetSpec.load(ROOT / "configs" / "dataset_phase4_v5.yaml")
    s = sample_designs(spec)[: a.n]
    old = {}
    for line in (
        (ROOT / "datasets" / "phase4_v5" / "designs.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        r = json.loads(line)
        if r["design_id"] < a.n:
            old[r["design_id"]] = r
    res = {"n_designs": a.n}
    for lab, k in (("v5_settings_k1", 1.0), ("v6_nominal_k1.86", 1.86)):
        cfg = load_engine_config()
        cfg["turbine_design"] = {**cfg["turbine_design"], "loss_multiplier": {"value": k}}
        new = Parallel(n_jobs=3)(delayed(evaluate_design)(d, cfg, spec.cfg["points"]) for d in s)
        res[lab] = compare(old, new)
        print(lab, yaml.safe_dump(res[lab], sort_keys=False), flush=True)
    out = ROOT / "validation" / "phase4" / "retry_recovery_check.yaml"
    out.write_text(yaml.safe_dump(res, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

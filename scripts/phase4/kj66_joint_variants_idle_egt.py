"""Idle-EGT check of the joint (geometry + k_loss) KJ-66 turbine fits (round 7 item 4, D-068).

Each joint-refit start (validation/phase4/turbine_joint_refit_<start>.yaml) is run in the KJ-66 replica of
scripts/phase4/kj66_replica_idle_egt.py (real compressor + R1, A9 fitted per nozzle run at the top speed) with its own
k_loss; idle (< 50 krpm) EGT error, idle TIT and top-speed thrust/SFC/turbine efficiency are reported, next to T-A
and T-B (round 6, validation/phase4/kj66_turbine_variants.yaml).

Usage:  uv run python scripts/phase4/kj66_joint_variants_idle_egt.py
Output: validation/phase4/kj66_joint_variants_idle_egt.yaml/.csv
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase4"


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


REP = _load("rep", "scripts/phase4/kj66_replica_idle_egt.py")
TC = _load("tcal", "scripts/phase4/turbine_calibration_murray.py")
VAR = _load("var", "scripts/phase4/kj66_turbine_variants.py")


def main() -> int:
    res, frames = {}, []
    orig = REP.turbine_params
    for start in ("near_T-A", "middle", "near_T-B"):
        f = OUT / f"turbine_joint_refit_{start}.yaml"
        if not f.exists():
            continue
        jp = yaml.safe_load(f.read_text(encoding="utf-8"))["params"]
        p = (jp["hub_tip"], jp["ngv_gauging_deg"], jp["rotor_gauging_deg"])
        _g, le = TC.build(p, k_loss=jp["k_loss"])
        REP.turbine_params = lambda v, p=p, le=le: (*p, float(le)) if v == "X" else orig(v)
        d, s = REP.run_all(label=f"joint {start}", variant="X", k_loss=jp["k_loss"])
        frames.append(d.assign(start=start))
        v = s.get("validation", {})
        res[start] = {"params": jp, "rotor_le_deg": round(float(le), 2),
                      "idle_lt_50krpm": {"EGT_err_K_mean": v.get("EGT_err_K_below_50krpm", {}).get("mean"),
                                         "TIT_K_mean": v.get("TIT_model_K_below_50krpm", {}).get("mean")},
                      "all_speeds": {"EGT_err_K_mean": v.get("EGT_err_K", {}).get("mean"), "EGT_err_K_rms": v.get("EGT_err_K", {}).get("rms"),
                                     "air_flow_err_pct_mean": v.get("m_err_pct", {}).get("mean"), "converged": v.get("converged")},
                      "run2_top_120krpm": VAR.at_speed(d, "Nozzle Run 2", 120.0) if not d.empty else {},
                      "runs_built": [r for r, x in s.items() if isinstance(x, dict) and "A9_fit_m2" in x]}  # fmt: skip
        print(start, res[start]["idle_lt_50krpm"], res[start]["all_speeds"], flush=True)
    REP.turbine_params = orig
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            OUT / "kj66_joint_variants_idle_egt.csv", index=False
        )
    res = json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (OUT / "kj66_joint_variants_idle_egt.yaml").write_text(
        yaml.safe_dump(res, sort_keys=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

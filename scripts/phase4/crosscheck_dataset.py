"""pyCycle-CEA cross-check of a generated dataset (CLAUDE.md rule): a random 2 % of the successfully designed
engines are re-run as 0-D design-point cycles in pyCycle-CEA with the component efficiencies the coupled model
produced; flag > 1 % on Fn or TSFC; the gate fails if > 5 % are flagged (then: stop and report before training).

Usage:  uv run python scripts/phase4/crosscheck_dataset.py
Output: validation/phase4/crosscheck_<name>.csv / .yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from turboml.gas import thermo
from turboml.physics import crosscheck
from turboml.physics.cycle_common import CycleInputs
from turboml.physics.engine import ComponentLosses, load_engine_config

ROOT = Path(__file__).resolve().parents[2]


def cycle_inputs(p: dict, L: ComponentLosses) -> CycleInputs:
    gas = thermo.products(p["far"])
    T5s = gas.T_isentropic(p["tit_K"], p["p05_Pa"] / p["p04_Pa"])
    eta_t = (gas.h(p["tit_K"]) - gas.h(p["T05_K"])) / (gas.h(p["tit_K"]) - gas.h(T5s))
    return CycleInputs(W_air_kg_s=p["W_air_kg_s"], pr_c=p["pr_c"], eta_c=p["eta_c"], tit_K=p["tit_K"], eta_t=eta_t,
                       eta_mech=L.eta_mech, eta_burner=L.eta_burner, dp_burner_frac=L.dp_burner_frac,
                       inlet_recovery=L.inlet_recovery, dp_nozzle_frac=L.dp_nozzle_frac, nozzle_cv=L.nozzle_cv,
                       lhv_J_kg=L.lhv_J_kg)  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset_phase4_v2.yaml"))
    args = ap.parse_args()
    dcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    recs = [
        json.loads(ln)
        for ln in (ROOT / dcfg["output_dir"] / "designs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    ok = sorted(
        (r for r in recs if r["status"] == "ok" and r["points"]["design"]["converged"]),
        key=lambda r: r["design_id"],
    )
    n = max(1, math.ceil(dcfg["crosscheck_fraction"] * len(ok)))
    rng = np.random.default_rng(dcfg["seed"] + 3)
    pick = [ok[i] for i in sorted(rng.choice(len(ok), size=n, replace=False))]
    L = ComponentLosses.from_config(load_engine_config())
    inputs = [cycle_inputs(r["points"]["design"], L) for r in pick]
    df = crosscheck.run(inputs, indices=[r["design_id"] for r in pick], n_jobs=-1)
    df["coupled_Fn_N"] = [r["points"]["design"]["Fn_N"] for r in pick]
    df["coupled_TSFC_kg_N_s"] = [r["points"]["design"]["TSFC_kg_N_s"] for r in pick]
    df["coupled_vs_pyc_Fn"] = df.coupled_Fn_N / df.Fn_N_pyc - 1
    df["coupled_vs_pyc_TSFC"] = df.coupled_TSFC_kg_N_s / df.TSFC_kg_N_s_pyc - 1
    df["flagged_coupled"] = (df.coupled_vs_pyc_Fn.abs() > crosscheck.FLAG_TOL_REL) | (
        df.coupled_vs_pyc_TSFC.abs() > crosscheck.FLAG_TOL_REL
    )
    g = crosscheck.gate(df)
    gate_coupled_frac = float(df.flagged_coupled.mean())
    out = ROOT / "validation" / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"crosscheck_{dcfg['name']}.csv", index=False)
    summ = {"designed_engines": len(ok), "checked": n, "fraction": dcfg["crosscheck_fraction"], "seed": dcfg["seed"] + 3,
            "simple_vs_pycycle": {"flagged": g.n_flagged, "flagged_frac": round(g.flagged_frac, 4), "gate_passed": g.passed},
            "coupled_vs_pycycle": {"flagged": int(df.flagged_coupled.sum()), "flagged_frac": round(gate_coupled_frac, 4),
                                   "gate_passed": gate_coupled_frac <= crosscheck.GATE_MAX_FLAGGED_FRAC,
                                   "max_abs_Fn_pct": round(100 * float(df.coupled_vs_pyc_Fn.abs().max()), 4),
                                   "max_abs_TSFC_pct": round(100 * float(df.coupled_vs_pyc_TSFC.abs().max()), 4)},
            "secondary_dFn_over_Fg_max_abs": round(float(df.secondary_dFn_over_Fg.abs().max()), 5),
            "design_ids": [r["design_id"] for r in pick]}  # fmt: skip
    (out / f"crosscheck_{dcfg['name']}.yaml").write_text(
        yaml.safe_dump(summ, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summ, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

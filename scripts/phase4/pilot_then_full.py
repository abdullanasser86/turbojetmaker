"""Run a dataset version's pilot, apply the stop rule, and continue to the full run only if it passes
(user decision 2026-10-25: pilot of `pilot_designs`; >= `min_feasible_fraction` feasible -> full run without waiting).

Usage:  uv run python scripts/phase4/pilot_then_full.py --config configs/dataset_phase4_v4.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def run(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    run(
        [
            "scripts/phase4/generate_dataset.py",
            "--config",
            a.config,
            "--limit",
            str(cfg["pilot_designs"]),
        ]
    )
    run(["scripts/phase4/pilot_feasibility_analysis.py", "--config", a.config])
    res = yaml.safe_load(
        (ROOT / "validation" / "phase4" / f"pilot_analysis_{cfg['name']}.yaml").read_text(
            encoding="utf-8"
        )
    )
    print(
        f"pilot feasible fraction {res['feasible_fraction_of_pilot']} (stop rule < {res['min_feasible_fraction']}): "
        f"{'STOP' if res['stop'] else 'continue to the full run'}",
        flush=True,
    )
    if res["stop"]:
        return 0
    run(["scripts/phase4/generate_dataset.py", "--config", a.config])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

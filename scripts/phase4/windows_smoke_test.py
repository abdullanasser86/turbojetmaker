"""Spawn-safety smoke test of the dataset generator (Windows has no fork: worker processes start fresh and import
everything again, so all work must sit behind `if __name__ == "__main__"` and be picklable).

Checks, on a handful of designs (one infeasible-by-construction, the rest from the active dataset config):
  1. serial evaluation;
  2. a `multiprocessing` pool with the "spawn" start method (what Windows uses) gives identical records;
  3. joblib/loky with 2 workers (the generator's backend) gives identical records;
  4. the real entry point `scripts/phase4/generate_dataset.py` runs with 2 workers into a temporary directory and
     produces the flat tables, and a second run resumes (skips everything).
Run on the Windows laptop before a long run:
    uv run python scripts/phase4/windows_smoke_test.py
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from joblib import Parallel, delayed

from turboml.data.generator import DatasetSpec, evaluate_design_json, sample_designs
from turboml.physics.engine import load_engine_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "dataset_phase4_v3.yaml"
N = 4


def _strip(line: str) -> dict:
    r = json.loads(line)
    r.pop("time_s", None)
    for p in r.get("points", {}).values():
        p.pop("time_s", None)
    return r


def main() -> int:
    t0 = time.time()
    spec = DatasetSpec.load(CONFIG)
    recs = sample_designs(spec)[:N]
    engine_cfg, points = load_engine_config(), spec.cfg["points"]
    serial = [_strip(evaluate_design_json(r, engine_cfg, points)) for r in recs]
    ok = True
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as ex:
        spawned = [
            _strip(x) for x in ex.map(evaluate_design_json, recs, [engine_cfg] * N, [points] * N)
        ]
    same = spawned == serial
    ok &= same
    print(f"spawn pool (2 workers) identical to serial: {same}")
    lok = [
        _strip(x)
        for x in Parallel(n_jobs=2, backend="loky")(
            delayed(evaluate_design_json)(r, engine_cfg, points) for r in recs
        )
    ]
    same = lok == serial
    ok &= same
    print(f"joblib/loky (2 workers) identical to serial: {same}")

    with tempfile.TemporaryDirectory() as td:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        name = f"smoke_{os.getpid()}"
        cfg.update(name=name, output_dir=str(Path(td) / name))
        cpath = Path(td) / "smoke.yaml"
        cpath.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        env = {**os.environ, "TURBOML_N_WORKERS": "2"}
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "phase4" / "generate_dataset.py"),
            "--config",
            str(cpath),
            "--limit",
            str(N),
        ]
        split = ROOT / "data" / "splits" / f"{name}_split.csv"
        try:
            r1 = subprocess.run(
                cmd, capture_output=True, text=True, env=env, cwd=ROOT, encoding="utf-8"
            )
            out = Path(td) / name
            n_lines = (
                len((out / "designs.jsonl").read_text(encoding="utf-8").splitlines())
                if (out / "designs.jsonl").exists()
                else 0
            )
            good = (
                r1.returncode == 0
                and n_lines == N
                and (out / "designs.csv").exists()
                and (out / "points.csv").exists()
            )
            print(
                f"generate_dataset.py with 2 workers: exit {r1.returncode}, {n_lines}/{N} records, tables written: {good}"
            )
            if r1.returncode:
                print(r1.stdout[-2000:], r1.stderr[-2000:])
            r2 = subprocess.run(
                cmd, capture_output=True, text=True, env=env, cwd=ROOT, encoding="utf-8"
            )
            resumed = r2.returncode == 0 and f"{N} done, 0 to run" in r2.stdout
            print(f"second run resumes (nothing re-evaluated): {resumed}")
            ok &= good and resumed
        finally:
            split.unlink(missing_ok=True)
    print(
        f"platform {sys.platform}, start method default {mp.get_start_method()}, {time.time() - t0:.0f} s -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Estimate the wall time of the full Phase 4 dataset run on THIS machine, without writing any data.

Evaluates a small, representative sample of designs (spread over the Sobol sequence) in parallel with the
configured worker count, measures the throughput, and extrapolates to n_designs.

Usage (on the laptop, plugged in, performance power mode):
    uv run python scripts/phase4/estimate_walltime.py              # ~4 designs per worker
    uv run python scripts/phase4/estimate_walltime.py --per-worker 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from joblib import Parallel, delayed

from turboml.compute import n_workers, physical_cores
from turboml.config import load_config
from turboml.data.generator import DatasetSpec, evaluate_design, sample_designs
from turboml.physics.engine import load_engine_config


def _timed(rec, engine_cfg, points):
    t = time.perf_counter()
    r = evaluate_design(rec, engine_cfg, points)
    return time.perf_counter() - t, r["status"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-worker", type=int, default=4)
    args = ap.parse_args()
    spec = DatasetSpec.load()
    workers = n_workers(load_config().compute.n_workers)
    designs = sample_designs(spec)
    k = max(workers * args.per_worker, 8)
    idx = np.linspace(0, len(designs) - 1, k).astype(int)  # spread over the whole sequence
    engine_cfg = load_engine_config()
    Parallel(n_jobs=workers)(
        delayed(lambda: None)() for _ in range(workers)
    )  # start the worker pool
    t0 = time.perf_counter()
    res = Parallel(n_jobs=workers)(
        delayed(_timed)(designs[i], engine_cfg, spec.cfg["points"]) for i in idx
    )
    wall = time.perf_counter() - t0
    cpu = np.array([r[0] for r in res])
    n = int(spec.cfg["n_designs"])
    est = wall / k * n
    print(f"physical cores {physical_cores()}, workers {workers}")
    print(
        f"sample: {k} designs, wall {wall:.1f} s, per-design CPU time mean {cpu.mean():.2f} s (median {np.median(cpu):.2f}, max {cpu.max():.1f})"
    )
    print(f"designs that failed at design mode in the sample: {sum(r[1] != 'ok' for r in res)}/{k}")
    print(f"ESTIMATE for {n} designs on this machine: {est / 60:.0f} min ({est / 3600:.2f} h)")
    print("(parallel efficiency and turbo boost under sustained load vary; plan for +30 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

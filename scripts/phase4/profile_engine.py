"""Phase 4 profiling: where the coupled model spends its time, and what the diffuser speedup bought.

The meanline compressor marches the vaneless diffuser with RK4. Each derivative evaluation needs the
meridional velocity from continuity (rho Cm A = mdot). The original code solved that with a bracketed root
finder from scratch every time; the speedup (`CentrifugalCompressor._solve_continuity_newton`) starts Newton
from the previous step's Cm and falls back to the bracketed solver if Newton fails. Same equations, same
answers; this script measures both paths on the baseline engine and checks they agree.

Usage:  uv run python scripts/phase4/profile_engine.py
Output: validation/phase4/profiling.yaml
"""

from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

import yaml

from turboml.physics import compressor as comp
from turboml.physics.engine import Engine, design_engine, load_engine_config

ROOT = Path(__file__).resolve().parents[2]
POINTS = [(1.0, 3000.0, 0.5), (0.8, 0.0, 0.0), (0.35, 0.0, 0.0)]  # the Phase 4 off-design points


def run(cfg: dict) -> tuple[dict, dict]:
    t = time.perf_counter()
    d = design_engine(cfg["baseline_design"], cfg)
    times = {"design_engine_s": time.perf_counter() - t}
    vals = {"design_Fn_N": d.design_point.Fn_N}
    eng = Engine(d, cfg)
    for f, alt, mach in POINTS:
        t = time.perf_counter()
        p = eng.off_design(f * d.design_point.n_rpm, alt, mach)
        key = f"n{int(round(f * 100))}_alt{int(alt)}_M{mach}"
        times[f"off_design_{key}_s"] = time.perf_counter() - t
        vals[f"{key}_Fn_N"] = p.Fn_N
    return times, vals


def main() -> int:
    cfg = load_engine_config()
    run(cfg)  # warm-up: builds and caches the compressor reference
    fast_t, fast_v = run(cfg)
    prof = cProfile.Profile()
    prof.enable()
    run(cfg)
    prof.disable()
    st = pstats.Stats(prof)
    total = st.total_tt
    top = sorted(st.stats.items(), key=lambda kv: kv[1][3], reverse=True)
    cum = {
        f"{Path(k[0]).name}:{k[2]}": round(v[3] / total, 3)
        for k, v in top[:25]
        if "turboml" in k[0]
    }

    orig = comp.CentrifugalCompressor._solve_continuity_newton
    comp.CentrifugalCompressor._solve_continuity_newton = (
        lambda self, *a, **k: None
    )  # old path only
    try:
        slow_t, slow_v = run(cfg)
    finally:
        comp.CentrifugalCompressor._solve_continuity_newton = orig
    out = {
        "machine_note": "measured in the cloud container (Xeon 2.1 GHz, one process); laptop timings differ",
        "wall_s_bracketed_solver": {k: round(v, 3) for k, v in slow_t.items()},
        "wall_s_newton_warm_start": {k: round(v, 3) for k, v in fast_t.items()},
        "speedup": {k: round(slow_t[k] / fast_t[k], 1) for k in fast_t},
        "max_rel_diff_outputs": float(max(abs(fast_v[k] / slow_v[k] - 1) for k in fast_v)),
        "cumulative_time_fraction_top_turboml_functions_after_speedup": cum,
    }
    path = ROOT / "validation" / "phase4" / "profiling.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(out, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

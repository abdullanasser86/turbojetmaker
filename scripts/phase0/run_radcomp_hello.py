"""Phase 0: RadComp hello-world.

1. Reproduces the single-point call from RadComp's EvaluateCompressor notebook
   (last entry of known_compressors.yml = Olmedo & Schiffmann stage 2, ammonia,
   0.12 kg/s, 130 krpm).
2. Runs a small speed sweep of the Schiffmann & Favrat (2010) impeller geometry with
   **air** at ISA sea-level inlet conditions, to show RadComp accepts air via CoolProp.
   That impeller is a ~20 mm heat-pump compressor, not a turbojet compressor -- these
   numbers are a plumbing check only, not a validation.

Usage:  uv run python scripts/phase0/run_radcomp_hello.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml
from radcompressor.compressor import Compressor
from radcompressor.condition import OperatingCondition
from radcompressor.geometry import Geometry
from radcompressor.thermo import CoolPropFluid

from turboml.units import ISA_SEA_LEVEL_P_PA, ISA_SEA_LEVEL_T_K, rpm_to_rad_s

DB = Path(__file__).resolve().parents[2] / "third_party" / "radcomp_known_compressors.yml"


def evaluate(geom: Geometry, fluid: str, p0_Pa: float, t0_K: float, m_kg_s: float, n_rpm: float):
    in0 = CoolPropFluid(fluid).thermo_prop("PT", p0_Pa, t0_K)
    op = OperatingCondition(in0=in0, fld=in0.fld, m=m_kg_s, n_rot=rpm_to_rad_s(n_rpm))
    comp = Compressor(geom, op)
    t = time.perf_counter()
    ok = comp.calculate()
    return ok, comp, (time.perf_counter() - t) * 1e3


def main() -> int:
    db = yaml.safe_load(DB.read_text())

    case = db[-1]
    c = case["conditions"]
    ok, comp, ms = evaluate(
        Geometry.from_dict(case["geom"]), c["fluid"], c["in_P"], c["in_T"], 0.120, 130e3
    )
    print(
        f"[notebook case] {case['name']} ({c['fluid']}): ok={ok} PR={comp.PR:.4f} "
        f"eta={comp.eff:.4f} ({ms:.0f} ms)"
    )

    sf = next(d for d in db if d["name"] == "Schiffmann and Favrat")
    geom = Geometry.from_dict(sf["geom"])
    print(
        f"\n[air sweep] {sf['name']} geometry, r4 = {geom.r4 * 1e3:.1f} mm, air at ISA SLS, m = 0.010 kg/s"
    )
    n_ok = 0
    for n_rpm in np.arange(120e3, 241e3, 20e3):
        ok, comp, ms = evaluate(geom, "Air", ISA_SEA_LEVEL_P_PA, ISA_SEA_LEVEL_T_K, 0.010, n_rpm)
        n_ok += bool(ok)
        flag = "ok " if ok else f"invalid ({comp.invalid_flag})"
        print(
            f"  N = {n_rpm / 1e3:5.0f} krpm  U2 = {rpm_to_rad_s(n_rpm) * geom.r4:5.0f} m/s  "
            f"PR = {comp.PR:6.3f}  eta = {comp.eff:6.3f}  {flag}  ({ms:.0f} ms)"
        )
    print(
        f"{n_ok} valid points. Invalid = RadComp flagged the point infeasible at this mass "
        "flow; the reason is not investigated in Phase 0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

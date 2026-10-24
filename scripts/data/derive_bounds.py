"""Statistics of the engine database used to set / check configs/bounds.yaml.

Usage:  uv run python scripts/data/derive_bounds.py
Reads validation/cases/review_engine_table.csv and jetcat_pro_catalog_2024.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "validation" / "cases"
MIN_THRUST_N = 50.0  # lower end of the TurboML thrust class


def main() -> int:
    d = pd.read_csv(CASES / "review_engine_table.csv")
    cat = pd.read_csv(CASES / "jetcat_pro_catalog_2024.csv")
    big = d[d.thrust_N >= MIN_THRUST_N]
    print(f"review table: {len(d)} engines, {len(big)} with thrust >= {MIN_THRUST_N:.0f} N")
    for col in ["pressure_ratio", "mass_flow_kg_s", "max_rpm", "diameter_mm", "max_egt_C"]:
        s = big[col].dropna()
        print(f"  {col:15s} n={len(s):3d}  min {s.min():9.3f}  max {s.max():9.3f}")
    st = (d.thrust_N / d.mass_flow_kg_s).dropna()
    print(f"  specific thrust [N s/kg]: n={len(st)} min {st.min():.0f} max {st.max():.0f}")
    small = d[(d.thrust_N < 250)]
    st_s = (small.thrust_N / small.mass_flow_kg_s).dropna()
    print(f"    engines < 250 N: {st_s.min():.0f}-{st_s.max():.0f}")
    m = d.dropna(subset=["max_rpm"])
    b, a = np.polyfit(np.log(m.thrust_N), np.log(m.max_rpm), 1)
    print(f"  fit max_rpm = {np.exp(a):.3g} * F^{b:.3f}: "
          f"{np.exp(a) * 50**b:.0f} rpm @ 50 N, {np.exp(a) * 1000**b:.0f} rpm @ 1000 N")  # fmt: skip
    print(
        f"catalog specific thrust [N s/kg]: {sorted((cat.thrust_N / cat.mass_flow_kg_s).round(0).unique())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

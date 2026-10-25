"""Reproduce the envelope/mass proxy coefficients in configs/mechanical.yaml from the engine databases.

Usage: uv run python scripts/phase3/fit_envelope.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    rev = pd.read_csv(ROOT / "validation" / "cases" / "review_engine_table.csv").dropna(
        subset=["diameter_mm", "length_mm"]
    )
    rev = rev[rev.thrust_N >= 50]
    ld = rev.length_mm / rev.diameter_mm
    print(
        f"L/D (review table, >= 50 N, n={len(ld)}): median {ld.median():.3f}, IQR {ld.quantile(.25):.3f}-{ld.quantile(.75):.3f}"
    )
    jc = pd.read_csv(ROOT / "validation" / "cases" / "jetcat_pro_catalog_2024.csv").dropna(
        subset=["mass_g", "diameter_mm", "length_mm"]
    )
    k = jc.mass_g / 1000 / ((jc.diameter_mm / 1000) ** 2 * (jc.length_mm / 1000))
    print(
        f"mass coefficient k = m/(D^2 L) (JetCat, n={len(k)}): mean {k.mean():.1f}, std {k.std():.1f}, range {k.min():.0f}-{k.max():.0f} kg/m3"
    )
    print(f"KJ-66 casing/impeller diameter: {53.7 / 33.0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

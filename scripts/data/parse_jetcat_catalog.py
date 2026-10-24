"""Extract the specification table of the JetCat PRO Engines catalog (2024, "JetCat 25Y").

Source: JetCat (Ingenieurbuero CAT, M. Zipperer GmbH), "JetCat PRO Engines" catalog 2024,
specification table on catalog pp. 14-15 (PDF page 8). Footnote: "all data at STP +/- 3%;
STP: 15 degC, 1013 mbar".

The PDF is not committed. Usage:
    uv run python scripts/data/parse_jetcat_catalog.py <path/to/JetCat_PROEngines_2024.pdf>
Output: validation/cases/jetcat_pro_catalog_2024.csv (one row per engine variant)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "cases" / "jetcat_pro_catalog_2024.csv"
CITATION = (
    "JetCat PRO Engines catalog 2024 (JetCat 25Y), spec table pp. 14-15; all data at STP +/-3%"
)

ROWS = {  # catalog row label -> (column name, unit conversion)
    "Idle rpm (1/min)": ("idle_rpm", 1.0),
    "Max rpm (1/min)": ("max_rpm", 1.0),
    "Thrust at idle (N)": ("idle_thrust_N", 1.0),
    "Maximum thrust (N)": ("thrust_N", 1.0),
    "Exhaust gas temperature (°C)": ("egt_C", None),
    "Pressure ratio": ("pressure_ratio", 1.0),
    "Mass flow (kg/s)": ("mass_flow_kg_s", 1.0),
    "Exhaust gas velocity (km/h)": ("exhaust_velocity_m_s", 1 / 3.6),
    "Exhaust gas power output (kW)": ("exhaust_power_kW", 1.0),
    "Fuel consumption idle (ml/min)": ("fuel_idle_ml_min", 1.0),
    "Fuel consumption @max rpm(ml/min)": ("fuel_max_ml_min", 1.0),
    "Fuel consumption idle (kg/min)": ("fuel_idle_kg_min", 1.0),
    "Fuel consumption @max rpm (kg/min)": ("fuel_max_kg_min", 1.0),
    "SFC @max rpm (kg/[N*h])": ("sfc_kg_N_h", 1.0),
    "Weight (g)": ("mass_g", 1.0),
    "Diameter (mm)": ("diameter_mm", 1.0),
    "Length (mm) incl. starter motor": ("length_mm", 1.0),
}


def parse(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    table: dict[str, dict] = {}
    names: list[str] = []
    for ln in lines:
        cells = [c for c in re.split(r"\s{2,}", ln.strip()) if c]
        if not cells:
            continue
        if cells[0] == "Typ":  # header: two side-by-side tables, each starting with "Typ"
            names = [c for c in cells if c != "Typ"]
            table = {n: {"variant": n} for n in names}
            continue
        label = cells[0]
        if label not in ROWS or not names:
            continue
        # the right-hand table repeats the label; drop the repeat
        values = [c for c in cells[1:] if c != label]
        if len(values) != len(names):
            raise ValueError(f"{label}: {len(values)} values for {len(names)} engines: {values}")
        col, scale = ROWS[label]
        for n, v in zip(names, values, strict=True):
            if scale is None:  # EGT range "480-750"
                lo, hi = (float(x) for x in v.split("-"))
                table[n]["egt_min_C"], table[n]["egt_max_C"] = lo, hi
            else:
                table[n][col] = float(v.replace(",", ".")) * scale
    df = pd.DataFrame(list(table.values()))
    df["citation"] = CITATION
    return df


def main() -> int:
    pdf = Path(sys.argv[1])
    text = subprocess.run(
        ["pdftotext", "-f", "8", "-l", "8", "-layout", str(pdf), "-"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    df = parse(text)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} variants -> {OUT.relative_to(ROOT)}")
    print(df.drop(columns="citation").round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

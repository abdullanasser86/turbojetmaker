"""Extract Table 3 (micro-turbojet specifications, <1000 N) from the review PDF.

Source: A. Anjomrouz, S. Ghadiri, A. Imani, "A review on the structures and characteristics
of micro-turbojet engines", J. Solid and Fluid Mechanics 13(5):59-76, 2023 (in Persian with
English abstract and tables). doi:10.22044/JSFM.2023.13523.3780. Table 3, pp. 71-74.

The PDF is not committed (copyright). Usage:
    uv run python scripts/data/parse_review_table.py <path/to/review.pdf>
Output: validation/cases/review_engine_table.csv

Every value is copied as printed. '-' or blank in the table becomes empty (NaN). The table's
own source column (manufacturer-website reference number) is kept per row, together with the
URL from the review's reference list.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "cases" / "review_engine_table.csv"
CITATION = (
    "Anjomrouz, Ghadiri & Imani (2023), J. Solid Fluid Mech. 13(5):59-76, "
    "doi:10.22044/JSFM.2023.13523.3780, Table 3"
)
# Review reference list entries cited in Table 3 (pp. 75-76).
REF_URL = {
    36: "https://www.hyblturbines.cz/", 52: "https://www.amtjets.com/",
    54: "Leylek et al. 2013, ASME GT2013 V05AT23A007", 55: "http://www.hawkturbine.com/",
    60: "http://www.jetjoe.com/", 61: "http://www.lambert-modellturbinen.de/",
    62: "https://www.jetcat.de/", 63: "http://www.kingtechturbines.com/",
    64: "https://www.turbinesolutions.co.uk/", 65: "http://www.xicoyturbines.com/",
    66: "https://www.pstjets.com/", 67: "https://www.sw-jet.com/",
    68: "https://jetcentral.com.mx/", 69: "https://jets-munt.com/",
    70: "https://jetpol.tech/", 71: "http://xtcm.evojets.de/",
    72: "http://www.bf-turbines.de/", 73: "http://www.behotec.de/",
    74: "https://frankturbine.com/", 75: "http://www.psr-jet-system.com/",
    76: "https://www.pbs.cz/en/", 77: "https://www.tei.com.tr/en",
    78: "https://propsan.com/", 79: "http://jetbeetle.com/",
    80: "https://www.polaristec.com.br/",
}  # fmt: skip
COLS = ["thrust_N", "diameter_mm", "length_mm", "max_rpm", "pressure_ratio",
        "mass_flow_kg_s", "fuel_g_min", "max_egt_C"]  # fmt: skip

# Rows whose name/company wrap over several text lines in the PDF.
NAME_FIXES = {
    "TS 45i": ("TS 45i", "Turbine Solutions"),
    "Wren 75": ("Wren 75", "Turbine Solutions"),
    "TS 100i": ("TS 100i", "Turbine Solutions"),
    "Wren 160": ("Wren 160", "Turbine Solutions"),
}
ORPHAN_ROWS = {  # numeric line without name -> (name, company)
    "80.4": ("Wren 80 Jubilee", "Turbine Solutions"),
    "JetCat 425": ("P400 PRO-LN", "JetCat"),
    "JetCat 550": ("P550 PRO-GL", "JetCat"),
}
COMPANIES = sorted(
    ["JetJoe", "Lambert", "JetCat", "KingTech", "XICOY", "PSTjets", "SWIWIN", "JetCentral",
     "AMT", "JetsMunt", "JetPol", "evoJet", "Hawk Turbine", "BF-Turbines", "Behotec",
     "Hybl Turbines", "Frank Turbine", "PSR", "PBS", "TEI", "PROPSAN", "Jetbeetle", "Polaris"],
    key=len, reverse=True,
)  # fmt: skip


def _num(tok: str) -> float | None:
    return None if tok in ("-", "") else float(tok)


def parse(text: str) -> pd.DataFrame:
    text = re.sub(r"[‎‏‪-‮⁦-⁩]", "", text)
    lines = [ln.strip() for ln in text.splitlines()]
    i0 = next(i for i, ln in enumerate(lines) if ln.startswith("J Junior"))
    i1 = next(i for i, ln in enumerate(lines) if ln.startswith("TJ200"))
    rows = []
    for ln in lines[i0 : i1 + 1]:
        m = re.search(r"(?:\]\[|\[)(\d+)\]?$", ln)
        if not m:
            continue
        ref = int(m.group(1))
        body = " ".join(ln[: m.start()].split())
        orphan = next((v for k, v in ORPHAN_ROWS.items() if body.startswith(k)), None)
        if orphan:
            name, company = orphan
            body = body[len(company) :].strip() if body.startswith(company) else body
        else:
            company = next(
                (c for c in COMPANIES if re.search(rf"\S\s+{re.escape(c)}\s", f" {body} ")), None
            )
            if company:
                name, body = (s.strip() for s in body.rsplit(company, 1))
            else:
                name = next((k for k in NAME_FIXES if body.startswith(k)), None)
                if name is None:
                    raise ValueError(f"unrecognised row: {ln!r}")
                company = NAME_FIXES[name][1]
                body = body[len(name) :].strip()
        toks = body.split()
        if name == "TJ90":  # diameter and length cells are blank in the PDF
            toks = [toks[0], "-", "-", *toks[1:]]
        if len(toks) != len(COLS):
            raise ValueError(f"cannot parse row {name!r}: {toks}")
        rows.append(
            {"name": name, "company": company, **{c: _num(t) for c, t in zip(COLS, toks, strict=True)},
             "table_ref": ref, "table_ref_source": REF_URL.get(ref, ""), "citation": CITATION}
        )  # fmt: skip
    return pd.DataFrame(rows)


def main() -> int:
    pdf = Path(sys.argv[1])
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout
    df = parse(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} engines -> {OUT.relative_to(ROOT)}")
    print(df[COLS].describe().T[["count", "min", "max"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

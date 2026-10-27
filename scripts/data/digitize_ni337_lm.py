"""Digitize the Larson-Miller curve of Nickel Institute publication No. 337, "Engineering Properties of Alloy 713C",
Figure 7 (p. 10): stress (1000 psi, log axis 1-100) vs P = T_R (20 + log10 t_h) x 1e-3 (T in degrees Rankine).

Method (as scripts/data/digitize_murray_ch5.py): 200-dpi render with pdftoppm; the plot frame is found from the long
dark lines (left/right frame = P 34 / 58, top/bottom frame = 100 / 1 ksi, cross-checked with the P gridlines every 2
and the 10 ksi line); the curve is traced column by column as the centroid of the dark run inside a +-25 px search
band around an eye-guided path (the band keeps the label text and the other curve out; amac points within the
label-arrow crossing, P 43.5-45.5, are dropped). Curves:
  vmvc  "Vacuum Melted Vacuum Cast" (solid to P ~ 52, dashed beyond: the dashed part is labelled `dashed_in_source`),
  amac  "Air Melted Argon Cast" (dashed over its whole length; a different melt practice, reference only).
Digitizing uncertainty: half the stroke width + 1 px, converted to P and stress.

The uploaded PDF is NOT committed (CLAUDE.md); pass its path:
    uv run python scripts/data/digitize_ni337_lm.py <path/to/alloys-713c_337.pdf>
Output: validation/cases/ni337_in713c_larson_miller_digitized.csv
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "cases" / "ni337_in713c_larson_miller_digitized.csv"
PAGE = 10
CROP = (700, 1720, 250, 1450)  # rows, cols of the lower (LM) chart on the 200-dpi page
# eye-guided paths (P, ksi) that only centre the search band; the values come from the pixels
EYE = {
    "vmvc": [
        (40.0, 100),
        (42, 72),
        (44, 53),
        (46, 39),
        (48, 28),
        (50, 19.5),
        (52, 13),
        (53, 10),
        (54, 7.5),
        (54.75, 6.0),
    ],
    "amac": [
        (39.8, 72),
        (42, 58),
        (44, 45),
        (46, 33),
        (48, 23),
        (50, 15.5),
        (51.5, 10),
        (52.5, 7.8),
    ],
}
BAND_PX = {"vmvc": 25, "amac": 12}  # amac: narrower, the label arrow crosses it near P 44-45


def render(pdf: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [
                "pdftoppm",
                "-r",
                "200",
                "-png",
                "-f",
                str(PAGE),
                "-l",
                str(PAGE),
                str(pdf),
                f"{td}/p",
            ],
            check=True,
        )
        png = next(Path(td).glob("p-*.png"))
        a = np.asarray(Image.open(png).convert("L"))
    r0, r1, c0, c1 = CROP
    return a[r0:r1, c0:c1] < 128


def frame(dark: np.ndarray) -> dict:
    rows, cols = dark.sum(1), dark.sum(0)
    long_r = np.flatnonzero(rows > 0.7 * dark.shape[1] * 0.83)
    long_c = np.flatnonzero(cols > 0.6 * dark.shape[0] * 0.8)
    top, bottom = long_r.min(), long_r.max()
    left, right = long_c.min(), long_c.max()
    # frame lines are 2-4 px thick: use the centre of the outermost runs
    top = float(np.mean([r for r in long_r if r - top < 5]))
    bottom = float(np.mean([r for r in long_r if bottom - r < 5]))
    left = float(np.mean([c for c in long_c if c - left < 5]))
    right = float(np.mean([c for c in long_c if right - c < 5]))
    return {
        "x0": left,
        "x1": right,
        "P0": 34.0,
        "P1": 58.0,
        "y_top": top,
        "y_bot": bottom,
        "lg_top": 2.0,
        "lg_bot": 0.0,
    }


def to_px(f, P, ksi):
    x = f["x0"] + (P - f["P0"]) / (f["P1"] - f["P0"]) * (f["x1"] - f["x0"])
    y = f["y_top"] + (f["lg_top"] - np.log10(ksi)) / (f["lg_top"] - f["lg_bot"]) * (
        f["y_bot"] - f["y_top"]
    )
    return x, y


def to_data(f, x, y):
    P = f["P0"] + (x - f["x0"]) / (f["x1"] - f["x0"]) * (f["P1"] - f["P0"])
    lg = f["lg_top"] - (y - f["y_top"]) / (f["y_bot"] - f["y_top"]) * (f["lg_top"] - f["lg_bot"])
    return P, 10**lg


def trace(dark, f, curve) -> pd.DataFrame:
    eye = np.array(EYE[curve], float)
    xs, ys = to_px(f, eye[:, 0], eye[:, 1])
    rows = []
    gridrows = set(
        np.flatnonzero(dark.sum(1) > 0.5 * dark.shape[1])
    )  # horizontal gridlines (full width)
    for x in range(int(np.ceil(xs[0])) + 2, int(xs[-1]) - 1, 4):
        yc = float(np.interp(x, xs, ys))
        lo, hi = int(yc - BAND_PX[curve]), int(yc + BAND_PX[curve])
        col = dark[lo:hi, x].copy()
        for r in range(lo, hi):
            if r in gridrows:
                col[r - lo] = False
        idx = np.flatnonzero(col)
        if idx.size < 2:
            continue  # gap in a dashed stroke or a gridline crossing
        # largest connected run in the band
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        if len(run) < 2:
            continue
        y = lo + float(run.mean())
        P, ksi = to_data(f, x, y)
        if curve == "amac" and 43.5 <= P <= 45.5:
            continue  # label arrow crosses the dashed curve here
        u_px = 0.5 * len(run) + 1
        rows.append({"curve": curve, "P_rankine_1e3": round(P, 3), "stress_ksi": round(ksi, 3),
                     "u_dig_stress_ksi": round(ksi * (10 ** (u_px / (f["y_bot"] - f["y_top"]) * 2) - 1), 3),
                     "u_dig_P": round(1.5 / (f["x1"] - f["x0"]) * 24, 3)})  # fmt: skip
    return pd.DataFrame(rows)


def main(pdf: Path) -> int:
    dark = render(pdf)
    f = frame(dark)
    # sanity: the 10 ksi gridline must sit half-way between the 100 and 1 ksi frame lines
    y10 = to_px(f, 46.0, 10.0)[1]
    r10 = np.flatnonzero(dark.sum(1) > 0.7 * dark.shape[1] * 0.83)
    assert np.min(np.abs(r10 - y10)) < 3, "frame calibration off (10 ksi line)"
    d = pd.concat([trace(dark, f, c) for c in ("vmvc", "amac")], ignore_index=True)
    d["P_K_1e3"] = (d.P_rankine_1e3 / 1.8).round(3)  # T_K = T_R / 1.8
    d["stress_MPa"] = (d.stress_ksi * 6.894757).round(2)
    d["segment"] = np.where(
        (d.curve == "vmvc") & (d.P_rankine_1e3 > 52.0),
        "dashed_in_source",
        np.where(d.curve == "amac", "dashed_style", "solid"),
    )
    d["data_type"] = "digitized [MEAS-derived curve]"
    header = [
        "# Nickel Institute No. 337, 'Engineering Properties of Alloy 713C', Figure 7 'Larson-Miller Stress Rupture Parameter",
        "# Curve' (p. 10). DIGITIZED by scripts/data/digitize_ni337_lm.py (200 dpi, curve centroid in a band around an eye path).",
        "# P_rankine_1e3 = T_R (20 + log10 t_h) / 1000 as printed (T_R = T_F + 460); P_K_1e3 = same with T in kelvin (T_R / 1.8).",
        "# vmvc = Vacuum Melted Vacuum Cast (the alloy of Tables II/III); amac = Air Melted Argon Cast (reference only).",
        "# segment: solid | dashed_in_source (vmvc beyond P ~ 52: dashed in the figure, treated as EXTRAPOLATED, not used for",
        "# allowables) | dashed_style (amac drawn dashed throughout). Typical (mean) properties, not design minima.",
    ]
    OUT.write_text(
        "\n".join(header) + "\n" + d.to_csv(index=False, lineterminator="\n"), encoding="utf-8"
    )
    print(f"frame {f}\n{len(d)} points -> {OUT}")
    print(
        d.groupby("curve").agg(
            P_min=("P_rankine_1e3", "min"),
            P_max=("P_rankine_1e3", "max"),
            n=("P_rankine_1e3", "size"),
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(Path(sys.argv[1])))

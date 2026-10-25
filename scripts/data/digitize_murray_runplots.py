"""Semi-automatic digitizer for Murray (2009) Appendix C run plots (Figs C.4-C.7): steady plateau values of the
traces "T5 * 0.1", "P03 (kPa)" and "(P01 - P1) * 100 (kPa)" at each set speed.

Method: the page is rendered at 200 dpi (pdftoppm). The y axis is calibrated from the horizontal gridlines (0, 10,
..., 130), the time axis from the plot frame (100 s and 1000 s, or the figure's own limits). For each plateau (time
window read from the speed steps, listed in PLATEAUS), the value of a trace is the median height of the pixels of that
trace's grey level inside a value band around a first by-eye estimate. Every value is DIGITIZED [MEAS]; the digitizing
uncertainty is estimated from Fig C.7, which is the same run as the Appendix D tables (see the validation step).

The uploaded PDF is NOT committed (copyright); this script and its output CSV are.
Usage: uv run python scripts/data/digitize_murray_runplots.py <murray.pdf>
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
OUT = ROOT / "validation" / "cases" / "murray2009" / "murray2009_runplot_plateaus_digitized.csv"

# figure -> (pdf page in the uploaded extract, thesis page, run, configuration, time axis: (t_left_s, px_left, s_per_px))
FIGS = {
    "C.4": (18, 128, "Nozzle Run 1, June 25 2009", "nozzle", (0.0, 362, 900.0 / 1054)),
    "C.5": (19, 129, "Nozzle Run 2, June 30 2009", "nozzle", (100.0, 362, 900.0 / 1048)),
    "C.6": (20, 130, "Nozzle Run 3, June 30 2009", "nozzle", (100.0, 362, 1.0 / 0.977)),
    "C.7": (
        21,
        131,
        "Vaneless Diffuser Run 3, June 30 2009",
        "exhaust_diffuser",
        (150.0, 362, 950.0 / 1048),
    ),
}
# plateaus: (set speed krpm, t0 s, t1 s, by-eye first estimates: T5*0.1, P03 gauge kPa, (P01-P1)*100)
PLATEAUS = {
    "C.4": [(48, 35, 75, 52.5, 17.5, 3.5), (56, 85, 125, 52.5, 24, 4.5), (64, 135, 180, 52.5, 31.5, 6.5),
            (72, 190, 230, 52.5, 40.5, 8.5), (80, 240, 285, 53, 51, 11), (80, 535, 585, 52.5, 50, 12.5),
            (88, 595, 640, 52.5, 61, 14), (96, 655, 695, 52.5, 74, 17), (104, 710, 750, 53, 89.5, 20),
            (112, 765, 800, 56, 104.5, 24.5)],
    "C.5": [(48, 130, 200, 49, 17.5, 3.8), (56, 212, 253, 48.5, 24.5, 5), (64, 264, 303, 48, 31.5, 7),
            (72, 314, 353, 48, 40.5, 9), (80, 364, 403, 49, 50.5, 11.5), (88, 414, 453, 49.5, 62.5, 14),
            (96, 464, 503, 50, 75.5, 17.5), (104, 517, 553, 50.5, 90.5, 20.5), (112, 569, 608, 52.5, 107, 24.5),
            (120, 624, 663, 53.5, 127, 29), (105.6, 677, 723, 51, 95, 21.5), (91.2, 735, 773, 48.5, 68.5, 15.5),
            (76.8, 785, 823, 48.5, 47, 10.5), (62.4, 837, 868, 49.5, 30, 6.5), (48, 884, 933, 50, 17.5, 4)],
    "C.6": [(48, 130, 175, 52.5, 17.5, 4), (56, 187, 228, 52, 24, 5.5), (64, 242, 278, 51.5, 31.5, 7),
            (72, 292, 328, 50, 40.5, 9.5), (80, 342, 373, 53, 50.5, 11), (88, 387, 423, 53, 62, 14),
            (96, 437, 473, 51.5, 75, 17.5), (104, 487, 528, 53, 90, 20.5), (112, 542, 578, 54.5, 107, 24.5),
            (120, 592, 633, 55.5, 125, 29), (104, 667, 718, 50.5, 90, 20.5), (88, 732, 773, 49.5, 62, 14),
            (72, 792, 843, 48.5, 40, 9), (56, 862, 913, 48, 24, 5.5), (48, 927, 973, 48.5, 18, 4),
            (32, 992, 1048, 54, 8, 1.5), (28, 1067, 1128, 55, 6, 1)],
    "C.7": [(48, 228, 268, 56, 17.5, 4), (56, 278, 318, 54.5, 24, 5.5), (64, 328, 368, 52, 31, 7.5),
            (72, 378, 418, 50, 40, 10), (80, 428, 468, 49, 50, 12.5), (88, 478, 518, 48, 62, 15.5),
            (96, 528, 568, 47, 75, 19.5), (104, 578, 618, 46.5, 90.5, 23), (112, 628, 673, 43, 105.5, 28),
            (120, 683, 738, 45, 123, 33), (105.6, 748, 788, 46.5, 91, 24.5), (91.2, 798, 843, 47.5, 64.5, 17),
            (76.8, 853, 888, 48, 45, 11.5), (62.4, 898, 943, 50, 29, 7), (48, 958, 1003, 53.5, 17, 4)],
}  # fmt: skip
Y0_PX, PX_PER_UNIT = None, None  # calibrated per page from the gridlines
BAND = {"T5": 4.0, "P03": 3.0, "dP": 2.0, "speed": 3.0}


def calibrate_y(g: np.ndarray) -> tuple[float, float, list[int]]:
    """Value-axis calibration from the long horizontal gridlines (0 ... 130 every 10 units)."""
    rows = np.where((g[:1400] < 128).sum(axis=1) > 800)[0]
    lines = [int(r) for r in rows if 230 <= r <= 1040]
    top, bottom = min(lines), max(lines)  # 130 and 0
    return float(bottom), (bottom - top) / 130.0, lines


def trace_value(g, x0, x1, est, band, kind, y0, ppu, grid):
    lo_px, hi_px = int(y0 - (est + band) * ppu), int(y0 - (est - band) * ppu)
    win = g[lo_px:hi_px, x0:x1]
    mask = (win >= 60) & (win <= 215) if kind == "T5" else (win < 60)
    for r in grid:  # remove gridline rows
        if lo_px - 2 <= r <= hi_px + 2:
            mask[max(r - lo_px - 1, 0) : max(r - lo_px + 2, 0), :] = False
    ys = np.nonzero(mask)[0]
    if ys.size < 10:
        return np.nan, 0
    return (y0 - (lo_px + np.median(ys))) / ppu, int(ys.size)


def main() -> int:
    pdf = Path(sys.argv[1])
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for fig, (page, thesis_page, run, config, (t_left, px_left, s_per_px)) in FIGS.items():
            subprocess.run(["pdftoppm", "-r", "200", "-png", "-f", str(page), "-l", str(page), str(pdf), f"{td}/p"],
                           check=True)  # fmt: skip
            png = next(Path(td).glob(f"p-{page:02d}.png"))
            g = np.array(Image.open(png).convert("L")).astype(int)
            y0, ppu, grid = calibrate_y(g)
            for sp, t0, t1, e_t5, e_p03, e_dp in PLATEAUS[fig]:
                x0, x1 = int(px_left + (t0 - t_left) / s_per_px), int(
                    px_left + (t1 - t_left) / s_per_px
                )
                t5, n5 = trace_value(g, x0, x1, e_t5, BAND["T5"], "T5", y0, ppu, grid)
                p03, n3 = trace_value(g, x0, x1, e_p03, BAND["P03"], "line", y0, ppu, grid)
                dp, nd = trace_value(g, x0, x1, e_dp, BAND["dP"], "line", y0, ppu, grid)
                spd, ns = trace_value(g, x0, x1, sp, BAND["speed"], "line", y0, ppu, grid)
                rows.append({"figure": fig, "thesis_page": thesis_page, "run": run, "configuration": config,
                             "set_speed_krpm": sp, "t0_s": t0, "t1_s": t1, "speed_krpm": round(spd, 1),
                             "T5_probe_C": round(10 * t5, 0), "P03_gauge_kPa": round(p03, 1),
                             "dP_inlet_kPa": round(dp / 100, 4), "pixels_T5": n5, "pixels_P03": n3, "pixels_dP": nd,
                             "eye_T5_probe_C": 10 * e_t5, "eye_P03_gauge_kPa": e_p03, "eye_dP_inlet_kPa": e_dp / 100})  # fmt: skip
    df = pd.DataFrame(rows)
    # digitizing error: Fig C.7 is the run tabulated in Appendix D (same plateau sequence, D.1 ... D.15)
    ad = pd.read_csv(OUT.parent / "murray2009_appD_running_data.csv", comment="#")
    c7 = df[df.figure == "C.7"].reset_index(drop=True)
    cp5 = 1110.0  # J/(kg K), Appendix D cp at station 5 is 1.089-1.135 kJ/(kg K); effect on the correction < 0.5 K
    ref = pd.DataFrame({"T5_probe_C": ad.T05_K - 0.32 * ad.V5_m_s**2 / (2 * cp5) - 273.15,
                        "P03_gauge_kPa": ad.P03_kPa - ad.P01_kPa})  # fmt: skip
    ref["dP_inlet_kPa"] = ad.P01_kPa - ad.Ps1_kPa
    err = c7[["T5_probe_C", "P03_gauge_kPa", "dP_inlet_kPa"]] - ref
    stats = {k: {"mean": round(float(err[k].mean()), 4), "rms": round(float((err[k] ** 2).mean() ** 0.5), 4),
                 "max_abs": round(float(err[k].abs().max()), 4)} for k in err}  # fmt: skip
    print("digitizer vs Appendix D (Fig C.7):", stats)
    unc = {"T5_probe_C": 2 * stats["T5_probe_C"]["rms"], "P03_gauge_kPa": 2 * stats["P03_gauge_kPa"]["rms"],
           "dP_inlet_kPa": 2 * stats["dP_inlet_kPa"]["rms"]}  # fmt: skip
    for k, v in unc.items():
        df[f"u_{k}"] = round(v, 4)
    df["data_type"] = "digitized [MEAS]"
    df["uncertainty_note"] = ("u_* = 2 x rms digitizing error of Fig C.7 vs the Appendix D tables of the same run "
                              "(includes plateau-window choice); T5 axis label '(K)' is wrong: values are deg C x 0.1")  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Murray (2009) MSc thesis, Appendix C Figs C.4-C.7 (thesis pp. 128-131): steady plateau values, DIGITIZED\n"
              f"# [MEAS] by scripts/data/digitize_murray_runplots.py from 200-dpi renders. T5_probe_C = raw exhaust thermocouple\n"
              f"# reading (the figures plot 'T5 * 0.1 (K)' but the values only match Appendix D in deg C; see VALIDATION.md).\n"
              f"# P03_gauge_kPa = compressor delivery total pressure above ambient; dP_inlet_kPa = P01 - P1 (bellmouth head).\n"
              f"# Digitizer error vs Appendix D (Fig C.7 = same run): {stats}\n")  # fmt: skip
    OUT.write_text(header + df.to_csv(index=False), encoding="utf-8")
    print(df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

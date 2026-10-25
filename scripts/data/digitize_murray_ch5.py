"""Digitize Murray (2009) Chapter 5 figures (user upload of printed pp. 75-93) into validation/cases/murray2009/.

Scatter maps (Figs 5.12 compressor PR, 5.14 compressor efficiency, 5.16 turbine corrected mass flow, 5.18 turbine
efficiency) -- semi-automatic:
  * 200-dpi render (pdftoppm); the plot frame (Excel plot area = axis ranges) is found from the long dark lines;
    gridlines are removed; the legend box is masked;
  * markers are connected dark components (lightly dilated so the strokes of x, *, + markers join); the centroid of
    each component is a point; overlapping markers merge into one component, flagged by `n_markers_est` (area /
    single-marker area) -- their centroid is the cluster mean;
  * speed class: compressor maps by the pressure-ratio band of each corrected-speed class (Fig 5.12), efficiency
    points by their corrected mass flow (same x as Fig 5.12); turbine maps by the pressure-ratio band of each turbine
    corrected-speed class (Fig 5.16, same x as Fig 5.18).
  * digitizing uncertainty = half the component size in data units (marker size / cluster spread), plus one pixel.
  * published uncertainty: the bands of Figs 5.13/5.15/5.17/5.19 are Murray's propagated uncertainties; their
    numbers are tabulated per running point in Appendix D, so the band half-width of each point is taken from the
    Appendix D point with the nearest corrected speed (u_pub_*). Spot checks against the plotted bands: see
    docs/VALIDATION.md.
Line plots (Figs 5.1 PR, 5.2 corrected mass flow, 5.3 turbine inlet temperature vs corrected speed): curve vertices
read by eye and snapped to the nearest dark curve pixel (EYE values in this file); Fig 5.3 "Vaneless Diffuser Run 3"
is the Appendix D run, which gives a measured digitizing error.

The uploaded PDF is NOT committed. Usage:
    uv run python scripts/data/digitize_murray_ch5.py <Murray_ch5.pdf>
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "cases" / "murray2009"
APPD = pd.read_csv(OUT / "murray2009_appD_running_data.csv", comment="#")

# figure: (pdf page, crop y0, crop y1 [px at 200 dpi], x range, y range, legend mask in plot fractions (x0, x1, y0, y1))
SCATTER = {
    "5.12": (14, 100, 845, (0.0, 0.25), (1.0, 2.4), (0.0, 0.42, 0.62, 1.0)),
    "5.14": (15, 100, 845, (0.0, 0.25), (0.5, 0.9), (0.6, 0.93, 0.07, 0.38)),
    "5.16": (17, 100, 845, (1.0, 2.0), (0.0, 0.18), (0.6, 0.9, 0.18, 0.55)),
    "5.18": (18, 100, 845, (1.0, 1.9), (0.5, 0.9), (0.6, 0.9, 0.07, 0.38)),
}
# corrected-speed classes and their pressure-ratio bands (Figs 5.12 and 5.16: the classes are separated in PR)
COMP_CLASSES = {40: (1.0, 1.16), 50: (1.16, 1.24), 60: (1.24, 1.33), 70: (1.33, 1.46), 80: (1.46, 1.6),
                90: (1.6, 1.77), 100: (1.77, 1.96), 110: (1.96, 2.18), 120: (2.18, 2.4)}  # fmt: skip
TURB_CLASSES = {20: (1.0, 1.09), 26: (1.09, 1.135), 32: (1.135, 1.19), 38: (1.19, 1.26), 44: (1.26, 1.35),
                50: (1.35, 1.46), 56: (1.46, 1.58), 62: (1.58, 1.705), 68: (1.705, 2.0)}  # fmt: skip


def render(pdf: Path, page: int, td: str) -> np.ndarray:
    subprocess.run(
        ["pdftoppm", "-r", "200", "-png", "-f", str(page), "-l", str(page), str(pdf), f"{td}/p"],
        check=True,
    )
    return np.array(Image.open(next(Path(td).glob(f"p-{page:02d}.png"))).convert("L")).astype(int)


def _groups(idx: np.ndarray) -> list[float]:
    if len(idx) == 0:
        return []
    return [float(np.mean(gp)) for gp in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)]


def calibrate(sub: np.ndarray) -> dict:
    """Axis calibration of one Excel chart: plot-area rows from the gridline family (first = y max, last = y min),
    the y axis and the right plot-area border as the long vertical lines just inside the outer chart border, and
    x from the tick marks below the x axis (equally spaced from x min to x max)."""
    dark = sub < 128
    rows = _groups(np.where(dark.sum(axis=1) > 0.40 * sub.shape[1])[0])
    top, bottom = rows[1], rows[-2]  # rows[0] / rows[-1] are the outer chart border
    span = dark[int(top) : int(bottom)].sum(axis=0)
    vcols = _groups(np.where(span > 0.9 * (bottom - top))[0])
    left, right = vcols[1], vcols[-2]  # vcols[0] / vcols[-1]: outer chart border
    below = dark[int(bottom) + 2 : int(bottom) + 8].sum(axis=0)
    ticks = [c for c in _groups(np.where(below >= 4)[0]) if left - 2 <= c <= right + 2]
    grid = [r for r in rows if top - 1 <= r <= bottom + 1]
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "xticks": ticks,
        "ygrid": grid,
    }


def scatter_points(g: np.ndarray, fig: str) -> pd.DataFrame:
    _, y0c, y1c, (xa, xb), (ya, yb), (lx0, lx1, ly0, ly1) = SCATTER[fig]
    sub = g[y0c:y1c]
    c = calibrate(sub)
    xt = np.array(c["xticks"])
    fx = np.polyfit(xt, np.linspace(xa, xb, len(xt)), 1)  # pixel column -> x
    yg = np.array(c["ygrid"])
    fy = np.polyfit(yg, np.linspace(yb, ya, len(yg)), 1)  # pixel row -> y
    lft, r, t, b = int(c["left"]) + 3, int(c["right"]) - 2, int(c["top"]) + 2, int(c["bottom"]) - 2
    inner = (sub[t:b, lft:r] < 128).copy()
    for row in yg:  # remove horizontal gridlines
        k = int(round(row)) - t
        inner[max(k - 1, 0) : max(k + 2, 0), :] = False
    h, w = inner.shape
    inner[int((1 - ly1) * h) : int((1 - ly0) * h), int(lx0 * w) : int(lx1 * w)] = (
        False  # legend box
    )
    lab, _ = ndimage.label(ndimage.binary_dilation(inner, iterations=2))
    rows_ = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = (lab[sl] == i) & inner[sl]
        area = int(comp.sum())
        hh, ww = comp.shape
        if area < 6 or hh > 120 or ww > 120:
            continue
        yy, xx = np.nonzero(comp)
        cx, cy = lft + sl[1].start + xx.mean(), t + sl[0].start + yy.mean()
        rows_.append({"x": np.polyval(fx, cx), "y": np.polyval(fy, cy), "area_px": area, "w_px": ww, "h_px": hh,
                      "u_dig_x": (ww / 2 + 1) * abs(fx[0]), "u_dig_y": (hh / 2 + 1) * abs(fy[0])})  # fmt: skip
    d = pd.DataFrame(rows_).sort_values("x").reset_index(drop=True)
    single = d.area_px.quantile(0.25)
    d["n_markers_est"] = np.maximum(1, np.round(d.area_px / single)).astype(int)
    return d


# line plots: (pdf page, crop rows, x range, y range, legend mask in plot fractions)
LINES = {
    "5.1": (2, (190, 975), (40.0, 130.0), (1.0, 2.5), (0.12, 0.6, 0.58, 1.0)),
    "5.2": (2, (1075, 1850), (40.0, 130.0), (0.0, 0.25), (0.06, 0.44, 0.58, 1.0)),
    "5.3": (3, (190, 960), (40.0, 130.0), (750.0, 1000.0), (0.09, 0.37, 0.64, 1.0)),
}
# Fig 5.3 vertices read BY EYE (corrected speed krpm, TIT K); snapped to the curve pixels below
FIG53_EYE = {
    "Nozzle Run 1": [(48, 819), (56, 823), (63, 825), (71, 832), (79, 850), (87, 856), (95, 867), (103, 880), (111, 922)],
    "Nozzle Run 2": [(48, 789), (56, 781), (63, 781), (71, 790), (79, 807), (87, 824), (95, 842), (103, 863), (111, 881), (119, 921)],
    "Nozzle Run 3": [(48, 814), (56, 818), (63, 820), (71, 821), (79, 850), (87, 855), (95, 861), (103, 879), (111, 899), (119, 940)],
    "Vaned Diffuser": [(40, 809), (48, 803), (56, 810), (63, 810), (71, 827), (79, 826), (87, 832), (95, 840), (103, 848), (111, 869), (117, 892), (125, 965)],
    "Vaneless Diffuser Run 1": [(48, 808), (56, 790), (63, 783), (71, 775), (79, 763), (87, 767), (95, 769), (103, 779), (111, 786), (118, 870)],
    "Vaneless Diffuser Run 2": [(48, 805), (56, 800), (63, 806), (71, 797), (79, 784), (87, 786), (95, 803)],
    "Vaneless Diffuser Run 3": [(48, 847), (56, 837), (63, 822), (71, 811), (79, 817), (87, 817), (95, 817), (103, 821), (111, 795), (119, 843)],
}  # fmt: skip


# gridline values from top to bottom (the line charts draw no line at the top of the y range)
LINE_GRID = {"5.1": [2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0], "5.2": [0.2, 0.15, 0.1, 0.05, 0.0],
             "5.3": [950.0, 900.0, 850.0, 800.0, 750.0]}  # fmt: skip


def line_frame(g: np.ndarray, fig: str):
    """Line charts: no right frame and no x ticks. x: y axis = x min, right end of the gridlines = x max;
    y: the gridlines with the values in LINE_GRID."""
    page, (r0, r1), (xa, xb), (ya, yb), legend = LINES[fig]
    sub = g[r0:r1]
    dark0 = sub < 128
    rows = _groups(np.where(dark0.sum(axis=1) > 0.40 * sub.shape[1])[0])
    grid = rows[1:-1]  # drop the outer chart border lines
    assert len(grid) == len(LINE_GRID[fig]), (fig, grid)
    bottom = grid[-1]
    span = dark0[int(grid[0]) : int(bottom)].sum(axis=0)
    left = _groups(np.where(span > 0.9 * (bottom - grid[0]))[0])[1]  # [0] is the outer border
    # right end of the plot: the longest continuous dark run along a gridline (the legend box can cut the top ones)
    runs = [int(np.argmin(dark0[int(round(gr)), int(left) :])) for gr in grid]
    right = left + max(runs) - 1
    fx = np.polyfit([xa, xb], [left, right], 1)  # x value -> pixel column
    fy = np.polyfit(LINE_GRID[fig], grid, 1)  # y value -> pixel row
    dark = sub < 160
    for row in grid:
        k = int(round(row))
        dark[k - 1 : k + 2, :] = False
    t, b, lft, r = int(np.polyval(fy, yb)) + 3, int(bottom) - 3, int(left) + 3, int(right) - 3
    mask = np.zeros_like(dark)
    mask[t:b, lft:r] = True
    lx0, lx1, ly0, ly1 = legend
    h, w = b - t, r - lft
    mask[
        t + int((1 - ly1) * h) : t + int((1 - ly0) * h), lft + int(lx0 * w) : lft + int(lx1 * w)
    ] = False
    return dark & mask, fx, fy


def envelope(g: np.ndarray, fig: str, xs) -> pd.DataFrame:
    """Min/max of all curve pixels at each x (Figs 5.1, 5.2: the seven runs overlap too closely to separate)."""
    dark, fx, fy = line_frame(g, fig)
    out = []
    for x in xs:
        c = int(round(np.polyval(fx, x)))
        rows = np.nonzero(dark[:, c - 2 : c + 3].any(axis=1))[0]
        if rows.size == 0:
            continue
        to_y = lambda row: (row - fy[1]) / fy[0]  # noqa: E731
        out.append(
            {
                "corrected_speed_krpm": x,
                "y_min": to_y(rows.max()),
                "y_max": to_y(rows.min()),
                "u_dig": 1.5 / abs(fy[0]),
            }
        )
    return pd.DataFrame(out)


def snap(g: np.ndarray, fig: str, pts, tol: float) -> list[tuple[float, float, float]]:
    """Move each by-eye (x, y) to the nearest curve pixel cluster in a +-tol window (column x +- 4 px)."""
    dark, fx, fy = line_frame(g, fig)
    out = []
    for x, y in pts:
        c, r = int(round(np.polyval(fx, x))), np.polyval(fy, y)
        lo, hi = int(r - tol * abs(fy[0])), int(r + tol * abs(fy[0]))
        rows = np.nonzero(dark[lo:hi, c - 4 : c + 5].any(axis=1))[0] + lo
        if rows.size == 0:
            out.append((x, float(y), float("nan")))
            continue
        groups = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
        best = min(groups, key=lambda gp: abs(gp.mean() - r))
        out.append(
            (x, float((best.mean() - fy[1]) / fy[0]), float(abs(best.mean() - r) / abs(fy[0])))
        )
    return out


def classify(v: float, classes: dict) -> int:
    for k, (lo, hi) in classes.items():
        if lo <= v < hi:
            return k
    return -1


def main() -> int:
    pdf = Path(sys.argv[1])
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for fig, spec in SCATTER.items():
            g = render(pdf, spec[0], td)
            res[fig] = scatter_points(g, fig)
    comp_pr, comp_eta = res["5.12"], res["5.14"]
    comp_pr["speed_class_krpm"] = [classify(v, COMP_CLASSES) for v in comp_pr.y]
    # efficiency points: speed class from the PR-map point with the nearest corrected mass flow
    comp_eta["speed_class_krpm"] = [
        int(comp_pr.loc[(comp_pr.x - v).abs().idxmin(), "speed_class_krpm"]) for v in comp_eta.x
    ]
    turb_m, turb_eta = res["5.16"], res["5.18"]
    turb_m["speed_class_krpm"] = [classify(v, TURB_CLASSES) for v in turb_m.x]
    turb_eta["speed_class_krpm"] = [classify(v, TURB_CLASSES) for v in turb_eta.x]

    # published uncertainty: Appendix D point with the nearest corrected speed (comp / turb)
    def pub(df, ncol, cols):
        out = {}
        for c_out, c_in in cols.items():
            out[c_out] = [
                float(APPD.loc[(APPD[ncol] - s).abs().idxmin(), c_in]) for s in df.speed_class_krpm
            ]
        return pd.DataFrame(out, index=df.index)

    comp_pr = pd.concat(
        [
            comp_pr,
            pub(
                comp_pr, "comp_Ncorr_krpm", {"u_pub_PR": "u_comp_PR", "u_pub_Mcorr": "u_m_air_kg_s"}
            ),
        ],
        axis=1,
    )
    comp_eta = pd.concat(
        [
            comp_eta,
            pub(
                comp_eta,
                "comp_Ncorr_krpm",
                {"u_pub_eta": "u_comp_eta", "u_pub_Mcorr": "u_m_air_kg_s"},
            ),
        ],
        axis=1,
    )
    turb_m = pd.concat(
        [
            turb_m,
            pub(
                turb_m, "turb_Ncorr_krpm", {"u_pub_PR": "u_turb_PR", "u_pub_Mcorr": "u_m_air_kg_s"}
            ),
        ],
        axis=1,
    )
    turb_eta = pd.concat(
        [
            turb_eta,
            pub(turb_eta, "turb_Ncorr_krpm", {"u_pub_eta": "u_turb_eta", "u_pub_PR": "u_turb_PR"}),
        ],
        axis=1,
    )
    names = {"5.12": ("comp", comp_pr, "Mcorr_kg_s", "PR"), "5.14": ("comp", comp_eta, "Mcorr_kg_s", "eta"),
             "5.16": ("turb", turb_m, "PR", "Mcorr_kg_s"), "5.18": ("turb", turb_eta, "PR", "eta")}  # fmt: skip
    frames = []
    for fig, (comp, df, xn, yn) in names.items():
        df = df.rename(
            columns={"x": xn, "y": yn, "u_dig_x": f"u_dig_{xn}", "u_dig_y": f"u_dig_{yn}"}
        )
        df.insert(0, "figure", fig)
        df.insert(1, "component", "compressor" if comp == "comp" else "turbine")
        df["data_type"] = "digitized [MEAS]"
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    head = ("# Murray (2009) MSc thesis, Chapter 5 measured turbomachinery maps, Figs 5.12/5.14 (compressor PR and\n"
            "# efficiency vs corrected mass flow) and 5.16/5.18 (turbine corrected mass flow and efficiency vs pressure ratio),\n"
            "# printed pp. 88-92. DIGITIZED [MEAS] by scripts/data/digitize_murray_ch5.py (marker centroids, 200 dpi).\n"
            "# Points from all runs (nozzle + vaned + vaneless diffusers), grouped by Murray in corrected-speed classes.\n"
            "# Corrected quantities as Murray (Table H.1): N sqrt(Tref/T0in), m sqrt(T0in/Tref) Pref/P0in (Tref ~288 K, Pref 101.325 kPa).\n"
            "# Turbine efficiency is DERIVED by Murray (T04 from an energy balance). u_dig_* = digitizing uncertainty (half the\n"
            "# marker/cluster size + 1 px); u_pub_* = published uncertainty (Appendix D value at the nearest corrected speed;\n"
            "# Figs 5.13/5.15/5.17/5.19 plot the same propagated uncertainties). n_markers_est > 1: overlapping markers merged.\n")  # fmt: skip
    (OUT / "murray2009_ch5_maps_digitized.csv").write_text(
        head + allp.round(5).to_csv(index=False), encoding="utf-8"
    )
    # line plots ----------------------------------------------------------------------------------------------
    rows = []
    with tempfile.TemporaryDirectory() as td:
        g2, g3 = render(pdf, 2, td), render(pdf, 3, td)
    for fig, g, yname in (("5.1", g2, "PR"), ("5.2", g2, "Mcorr_kg_s")):
        env = envelope(g, fig, np.arange(48, 121, 4.0))
        for r in env.itertuples():
            rows.append({"figure": fig, "series": "all 7 runs (envelope)", "corrected_speed_krpm": r.corrected_speed_krpm,
                         "quantity": yname, "value_min": r.y_min, "value_max": r.y_max, "u_dig": r.u_dig})  # fmt: skip
    for run, pts in FIG53_EYE.items():
        for x, y, moved in snap(g3, "5.3", pts, tol=8.0):
            rows.append({"figure": "5.3", "series": run, "corrected_speed_krpm": x, "quantity": "TIT_K (calculated by Murray)",
                         "value_min": y, "value_max": y, "u_dig": 5.0, "snap_shift": moved})  # fmt: skip
    lines = pd.DataFrame(rows)
    lines["data_type"] = "digitized [MEAS]"
    # digitizing error of Fig 5.3: Vaneless Diffuser Run 3 = Appendix D T04 at the same corrected speeds
    v3 = lines[(lines.figure == "5.3") & (lines.series == "Vaneless Diffuser Run 3")].reset_index(
        drop=True
    )
    ref = APPD  # Appendix D order = run order (D.1 ... D.10 rising speed)
    err = v3.value_min.values - ref.T04_K.values[: len(v3)]
    lines.attrs["fig53_error"] = err
    head2 = (
        "\n".join(
            [
                "# Murray (2009) Chapter 5, Figs 5.1 (engine PR), 5.2 (corrected mass flow), 5.3 (turbine inlet temperature) vs",
                "# compressor corrected speed, printed pp. 76-77. DIGITIZED [MEAS] by scripts/data/digitize_murray_ch5.py.",
                "# Figs 5.1/5.2: the 7 runs (3 nozzle, vaned diffuser, 3 vaneless diffuser) overlap within the line width, so the",
                "# ENVELOPE of all curve pixels (min/max) is stored. Fig 5.3: per-run vertices read by eye and snapped to the curve",
                "# pixels (snap_shift = K moved); TIT is CALCULATED by Murray (T05 + turbine work), not measured.",
                f"# Fig 5.3 digitizing check: 'Vaneless Diffuser Run 3' vs Appendix D T04 (same run): mean {err.mean():+.1f} K, "
                f"rms {float(np.sqrt((err**2).mean())):.1f} K, max |{float(np.abs(err).max()):.1f}| K -> u_dig 5 K.",
            ]
        )
        + "\n"
    )
    (OUT / "murray2009_ch5_operating_lines_digitized.csv").write_text(
        head2 + lines.round(4).to_csv(index=False), encoding="utf-8"
    )
    print("Fig 5.3 check vs Appendix D T04 (K):", np.round(err, 1))
    for fig, df in zip(names, frames, strict=True):
        print(
            fig,
            len(df),
            "points; classes:",
            df.speed_class_krpm.value_counts().sort_index().to_dict(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

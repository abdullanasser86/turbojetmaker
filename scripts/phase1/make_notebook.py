"""Build notebooks/01_cycle_model.ipynb (execute afterwards with nbconvert --execute).

Requires the outputs of scripts/phase1/cross_check_sweep.py and jetcat_inverse.py.
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        "# Phase 1: the cycle model\n\n"
        "A **cycle model** follows the air through the engine (intake → compressor → combustor → "
        "turbine → nozzle) and applies conservation of mass and energy plus one efficiency per "
        "component. It returns thrust, fuel flow, and the temperature and pressure at every "
        "station.\n\n"
        "We built two of them:\n"
        "* `cycle_pycycle`: NASA's pyCycle, adapted to a micro turbojet;\n"
        "* `cycle_simple`: our own independent 0-D model (≈5 ms per run instead of ≈1 s).\n\n"
        "When two independent models agree under the same assumptions, a bug in either one "
        "is unlikely. That's *verification*. Agreement with real engines is a separate question; "
        "section 4 covers it."
    ),
    code(
        "import os, warnings\n"
        "from dataclasses import replace\n"
        "from pathlib import Path\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd(); os.chdir(ROOT)\n"
        "from turboml.physics.cycle_common import CycleInputs\n"
        "from turboml.physics.cycle_simple import design_point\n"
        "warnings.filterwarnings('ignore')\n"
        "SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']  # fixed categorical order\n"
        "INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dd'\n"
        "plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,\n"
        "    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,\n"
        "    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2, 'font.size': 10})"
    ),
    md(
        "## 1. One engine, station by station\n"
        "These are the baseline inputs from `configs/cycle_baseline.yaml`, with the pressure ratio and "
        "mass flow of a JetCat P160. The **component efficiencies are placeholders**; the YAML marks "
        "each value as an assumption, a datasheet value or a spec."
    ),
    code(
        "x = CycleInputs.from_yaml()\n"
        "r = design_point(x)\n"
        "print(f'Fn = {r.Fn_N:.1f} N   TSFC = {r.TSFC_kg_N_s*3600:.3f} kg/(N h) = {r.TSFC_g_kN_s:.1f} g/(kN s)   '\n"
        "      f'FAR = {r.far:.4f}   EGT = {r.EGT_K-273.15:.0f} C   nozzle choked: {r.choked}')\n"
        "names = {'0': 'free stream', '2': 'compressor in', '3': 'compressor out', '4': 'turbine in',\n"
        "         '5': 'turbine out', '9': 'nozzle exit'}\n"
        "pd.DataFrame({'station': [names[k] for k in r.stations],\n"
        "              'Tt [K]': [s.Tt_K for s in r.stations.values()],\n"
        "              'Pt [kPa]': [s.Pt_Pa / 1e3 for s in r.stations.values()]}, index=list(r.stations)).round(1)"
    ),
    md(
        "## 2. Verification: pyCycle vs the 0-D model\n"
        "We ran 64 Sobol-sampled designs through both models with identical inputs (ranges in "
        "`configs/phase1_crosscheck.yaml`: ground to 6 km, Mach 0–0.6, wide efficiency ranges). "
        "pyCycle ran with both of its gas-property options: CEA (equilibrium chemistry) and "
        "TABULAR (pre-computed tables, ~3× faster)."
    ),
    code(
        "cc = pd.read_csv('validation/phase1/cross_check.csv')\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)\n"
        "for ax, q, label in zip(axes, ['Fn_N', 'TSFC_kg_N_s'], ['net thrust', 'TSFC']):\n"
        "    for i, th in enumerate(['CEA', 'TABULAR']):\n"
        "        g = cc[cc.thermo == th]\n"
        "        ax.scatter(g.Fn_N_simple / g.W_air_kg_s, 100 * g[f'{q}_reldiff'], s=22, color=SERIES[i],\n"
        "                   edgecolor='white', linewidth=0.8, label=f'pyCycle {th}', zorder=3)\n"
        "    ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=1, lw=0)\n"
        "    ax.set(xlabel='specific thrust Fn / W  [N per kg/s]', title=f'{label}: pyCycle vs 0-D')\n"
        "axes[0].set_ylabel('relative difference [%]'); axes[0].legend(frameon=False)\n"
        "axes[1].annotate('±1 % band', xy=(60, 1.15), color=MUTED, fontsize=9)\n"
        "fig.tight_layout()\n"
        "cc.groupby('thermo')[['Fn_N_reldiff', 'TSFC_kg_N_s_reldiff', 'P_comp_W_reldiff']].agg(\n"
        "    lambda s: f'median {s.abs().median():.3%} / max {s.abs().max():.3%}')"
    ),
    md(
        "**How to read this:**\n"
        "* **CEA vs 0-D**: median differences are a few hundredths of a percent. The one point above "
        "1 % is an engine with almost no net thrust (specific thrust ≈ 36), where thrust is a small "
        "difference between two large numbers (jet momentum − ram drag), so any tiny difference is "
        "magnified.\n"
        "* **TABULAR vs 0-D**: errors are up to ~3 % because pyCycle's property tables are interpolated. "
        "Compressor work alone differs by up to 1 %, even though CEA and the 0-D model agree on it to "
        "machine precision. **Conclusion: don't use TABULAR for training data.**"
    ),
    md(
        "## 3. Trends: what the cycle wants\n"
        "Classic cycle behaviour, computed with the fast 0-D model. Higher pressure ratio cuts TSFC. "
        "Higher TIT raises specific thrust, so the engine gets smaller for the same thrust."
    ),
    code(
        "prs = np.linspace(1.8, 5.0, 33)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))\n"
        "for i, tit in enumerate([1000, 1100, 1200, 1300]):\n"
        "    res = [design_point(replace(x, pr_c=p, tit_K=tit)) for p in prs]\n"
        "    ok = [rr.converged for rr in res]\n"
        "    axes[0].plot(prs[ok], [rr.TSFC_kg_N_s * 3600 for rr, k in zip(res, ok) if k], color=SERIES[i])\n"
        "    axes[1].plot(prs[ok], [rr.specific_thrust_N_s_kg for rr, k in zip(res, ok) if k], color=SERIES[i],\n"
        "                 label=f'TIT {tit} K')\n"
        "    axes[1].annotate(f'{tit} K', (prs[ok][-1], [rr.specific_thrust_N_s_kg for rr, k in zip(res, ok) if k][-1]),\n"
        "                     xytext=(4, 0), textcoords='offset points', va='center', color=INK, fontsize=9)\n"
        "axes[0].set(xlabel='compressor pressure ratio', ylabel='TSFC [kg/(N h)]', title='fuel consumption')\n"
        "axes[1].set(xlabel='compressor pressure ratio', ylabel='Fn / W [N per kg/s]', title='specific thrust')\n"
        "axes[1].legend(frameon=False, loc='upper left'); fig.tight_layout()"
    ),
    md(
        "## 4. Sanity check against real engines (JetCat datasheets)\n"
        "Five JetCat engines from 63 to 425 N publish pressure ratio, mass flow, thrust and SFC "
        "(`validation/cases/jetcat_datasheets.yaml`). **These values came from web-search extracts and "
        "are unverified.** \n\n"
        "**Forward:** placeholder efficiencies + the datasheet PR and mass flow → predicted thrust "
        "and SFC."
    ),
    code(
        "fwd = pd.read_csv('validation/phase1/jetcat_forward.csv')\n"
        "fwd.assign(thrust_err=lambda d: (100 * d.thrust_err).round(1), sfc_err=lambda d: (100 * d.sfc_err).round(1))[\n"
        "    ['engine', 'thrust_datasheet_N', 'thrust_model_N', 'thrust_err', 'sfc_datasheet', 'sfc_model', 'sfc_err']].round(3)"
    ),
    md(
        "SFC is under-predicted by 10–15 % for every engine, so the placeholder efficiencies are too "
        "optimistic for real micro engines.\n\n"
        "**Inverse:** thrust and SFC give two equations, so for each assumed compressor efficiency η_c "
        "we solve for the TIT and turbine efficiency η_t that reproduce the datasheet *exactly*. Then we "
        "ask whether those implied values are physically plausible."
    ),
    code(
        "inv = pd.read_csv('validation/phase1/jetcat_inverse.csv')\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))\n"
        "for i, (name, g) in enumerate(inv[inv.eta_burner == 0.95].groupby('engine', sort=False)):\n"
        "    axes[0].plot(g.eta_c, g.eta_t, color=SERIES[i], label=name)\n"
        "egt = inv[inv.eta_c == 0.70].pivot(index='engine', columns='eta_burner', values='egt_C')\n"
        "egt = egt.loc[inv.engine.unique()]\n"
        "for j, eb in enumerate(egt.columns):\n"
        "    axes[1].bar(np.arange(len(egt)) + (j - 1) * 0.26, egt[eb], width=0.24, color=SERIES[j],\n"
        "                edgecolor='white', linewidth=2, label=f'η_b = {eb:.2f}', zorder=3)\n"
        "axes[1].axhline(750, color=INK, lw=1, ls='--', label='listed EGT max 750 °C (P62: 730)')\n"
        "axes[0].set(xlabel='assumed compressor efficiency η_c', ylabel='implied turbine efficiency η_t',\n"
        "            title='component efficiencies consistent with the datasheet')\n"
        "axes[0].legend(frameon=False, fontsize=8)\n"
        "axes[1].set_xticks(range(len(egt)), egt.index, rotation=20)\n"
        "axes[1].set(ylabel='implied EGT [°C]', ylim=(500, 1100), title='implied exhaust temperature')\n"
        "axes[1].legend(frameon=False, fontsize=8, loc='upper left'); fig.tight_layout()"
    ),
    md(
        "**What this tells us:**\n"
        "* By energy conservation, the exhaust temperature depends almost only on the fuel–air "
        "ratio, and fuel–air ratio = thrust × SFC / mass flow. So the implied EGT is "
        "nearly independent of the assumed component efficiencies.\n"
        "* **P160** comes out consistent for any combustion efficiency, and **P62** if η_b ≤ 0.93: the implied "
        "EGT is then within the listed range, with η_t of 0.62–0.77 depending on η_c.\n"
        "* **P300 and P400** come out inconsistent. Their numbers would need EGT ≈ 930–980 °C at η_b = 0.95, "
        "or a combustion efficiency of ≤ 0.70–0.74 to stay at the listed 750 °C. Either the extracted "
        "mass flow or thrust is wrong or measured differently, or the listed EGT isn't the mean exhaust "
        "total temperature (thermocouple placement). **We can't tell which without the official datasheets.**"
    ),
    md(
        "## 5. A pitfall we found in pyCycle: fuel enthalpy\n"
        "pyCycle's combustor has a fuel-enthalpy input that defaults to **0**, and its examples never set "
        "it. With pyCycle's thermo data, that makes C12H23 'kerosene' release ≈ 44.8 MJ/kg instead of "
        "~43. Our wrapper sets it from the LHV. Here's the size of the effect (0-D model):"
    ),
    code(
        "from turboml.gas.thermo import fuel_enthalpy_J_per_kg\n"
        "lhv_default = -fuel_enthalpy_J_per_kg(0.0)  # LHV implied by h_fuel = 0\n"
        "a = design_point(replace(x, eta_burner=1.0, lhv_J_kg=42.8e6))\n"
        "b = design_point(replace(x, eta_burner=1.0, lhv_J_kg=lhv_default))\n"
        "print(f'implied LHV with pyCycle default: {lhv_default/1e6:.2f} MJ/kg')\n"
        "print(f'TSFC with LHV 42.8 MJ/kg: {a.TSFC_kg_N_s*3600:.4f};  with the pyCycle default: '\n"
        "      f'{b.TSFC_kg_N_s*3600:.4f} kg/(N h)  ->  {100*(b.TSFC_kg_N_s/a.TSFC_kg_N_s-1):.1f} %')"
    ),
    md(
        "## Summary\n"
        "* Two independent cycle models agree to ~0.1 % (pyCycle CEA vs 0-D), so verification passes "
        "except at one ill-conditioned point.\n"
        "* The model can reproduce published thrust and SFC of the small JetCat engines with plausible "
        "component efficiencies. The larger PRO engines' extracted data looks inconsistent and needs "
        "checking.\n"
        "* Component efficiencies are still **assumptions**. Sourcing them is the most important open "
        "item. See `docs/PHASE1_REPORT.md` for the decisions needed."
    ),
]

nb = nbf.v4.new_notebook(
    cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}}
)
out = Path(__file__).resolve().parents[2] / "notebooks" / "01_cycle_model.ipynb"
nbf.write(nb, out)
print("wrote", out)

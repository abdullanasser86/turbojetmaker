"""Build notebooks/03_coupled_engine.ipynb (execute with nbconvert --execute afterwards).

Needs the outputs of scripts/phase3/kofskey_fast_turbine.py, turbine_port_verification.py and
baseline_engine.py.
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        "# Phase 3: the coupled engine model\n\n"
        "Phases 1 and 2 built the pieces: a cycle model that needs component efficiencies *assumed*, and "
        "meanline models that *compute* a compressor's and a turbine's performance from geometry. Phase 3 "
        "couples them on one shaft. At each shaft speed the engine settles where three conditions hold "
        "together (*matching*):\n\n"
        "1. **power balance**: the turbine drives the compressor, $\\eta_m P_t = P_c$;\n"
        "2. **turbine flow capacity**: the turbine swallows exactly the air plus fuel the compressor delivers;\n"
        "3. **nozzle flow capacity**: the fixed nozzle area passes that flow.\n\n"
        "Unknowns are air mass flow, turbine inlet temperature (TIT, i.e. fuel flow) and turbine pressure "
        "ratio; a Newton solver finds them. To make this fast enough we first wrote our own **fast axial "
        "turbine model** (TurboFlow's equations, marching solver) and verified it against TurboFlow.\n\n"
        "> Preliminary design only, not manufacturing-ready: everything here is a meanline/0-D model."
    ),
    code(
        "import os, warnings\n"
        "from pathlib import Path\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml\n"
        "from IPython.display import Markdown, display\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd(); os.chdir(ROOT)\n"
        "warnings.filterwarnings('ignore')\n"
        "SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']\n"
        "INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dd'\n"
        "plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,\n"
        "    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,\n"
        "    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2, 'font.size': 10})\n"
        "P3 = ROOT / 'validation' / 'phase3'"
    ),
    md(
        "## 1. Fast turbine model vs TurboFlow and vs measurement (Kofskey 1972)\n"
        "Same equations as TurboFlow (Benner losses, Aungier deviation, critical-Mach choking), solved plane "
        "by plane instead of all at once. Lines: our model; open circles: TurboFlow; filled dots: NASA "
        "measurements. The lines sit on the circles (port verification); both differ from the measurements by "
        "the same model error (validation)."
    ),
    code(
        "fm = pd.read_csv(P3 / 'kofskey_fast_map.csv'); tf = pd.read_csv('validation/phase2/kofskey_map.csv')\n"
        "meas = pd.read_csv('validation/cases/kofskey1972_turbine_measured.csv', comment='#')\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))\n"
        "for i, sp in enumerate([50, 70, 90, 100, 110]):\n"
        "    f = fm[(fm.incidence_sign == 'turboflow') & (fm.speed_pct_design == sp) & fm.converged]\n"
        "    t = tf[(tf.speed_pct_design == sp) & tf.converged]\n"
        "    for ax, col, q, sc in [(axes[0], 'mass_flow_kg_s', 'mass_flow', 1), (axes[1], 'eta_ts', 'efficiency_ts', 100)]:\n"
        "        ax.plot(f.pr_ts, f[col] * sc, color=SERIES[i], lw=1.6, label=f'{sp} %')\n"
        "        ax.scatter(t.pr_ts[::4], t[col][::4] * sc, s=22, facecolor='none', edgecolor=SERIES[i])\n"
        "        m = meas[(meas.speed_pct_design == sp) & (meas.quantity == q)]\n"
        "        ax.scatter(m.pressure_ratio_ts, m.value, s=26, color=SERIES[i], edgecolor=INK, linewidth=0.5, zorder=4)\n"
        "axes[0].set(xlabel='total-to-static pressure ratio', ylabel='mass flow [kg/s]'); axes[1].set(xlabel='total-to-static pressure ratio', ylabel='eta_ts [%]')\n"
        "axes[0].legend(title='speed', frameon=False, fontsize=8); plt.tight_layout(); plt.show()\n"
        "S = yaml.safe_load((P3 / 'kofskey_fast_summary.yaml').read_text())\n"
        "pd.DataFrame({m: {**S[m]['vs_turboflow_map'], 'eta_vs_meas_mean_pts': S[m]['vs_measured']['efficiency_ts']['mean_pts'],\n"
        "              'mdot_vs_meas_mean_%': S[m]['vs_measured']['mass_flow']['mean_%'], 'time_ms': S[m]['time_ms_median_sweep']}\n"
        "              for m in ('turboflow', 'rotor_plus')})"
    ),
    md(
        "**The incidence-sign question.** TurboFlow's code intends a different incidence sign for rotors "
        "(`rotor_plus`) but, through a small bug, uses the stator sign everywhere. Surprisingly, the buggy "
        "version fits the measurements *better* (at 50 % speed the rotor sees +27 deg incidence, and the "
        "measured efficiency there is not penalised as much as the intended sign predicts). We keep the "
        "TurboFlow behaviour as default and ask for the Benner et al. (1997) paper to settle it."
    ),
    md(
        "## 2. Verification across the design space (Sobol samples)\n"
        "Stages sized from random-but-even (Sobol) choices of flow coefficient, loading, reaction, aspect "
        "ratio, clearance, work, mass flow, speed, inlet temperature/pressure, and an off-design point. "
        "Acceptance: efficiency within 2 points, mass flow and work within 2 %."
    ),
    code(
        "sob = pd.read_csv(P3 / 'turbine_port_sobol.csv'); SS = yaml.safe_load((P3 / 'turbine_port_summary.yaml').read_text())\n"
        "c = sob[sob.status == 'compared']\n"
        "fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))\n"
        "for ax, col, lab in zip(axes, ['d_eta_ts_pts', 'd_mdot_pct', 'd_work_pct'], ['eta_ts [pts]', 'mass flow [%]', 'work [%]']):\n"
        "    ax.hist(c[col], bins=40, color=SERIES[0]); ax.axvline(-2, color=SERIES[1], ls='--'); ax.axvline(2, color=SERIES[1], ls='--')\n"
        "    ax.set(xlabel=f'fast - TurboFlow: {lab}', ylabel='points')\n"
        "plt.tight_layout(); plt.show()\n"
        "print(yaml.safe_dump({k: SS[k] for k in ('samples', 'points', 'status_counts', 'accepted', 'fast_time_ms_median_cold', 'tf_time_s_median')}, sort_keys=False))\n"
        "pd.DataFrame(SS['worst_cases'])"
    ),
    md(
        "## 3. The baseline engine (design card)\nA ~150 N engine sized by the design mode. All numbers are physics-evaluated."
    ),
    code("display(Markdown((P3 / 'baseline_design_card.md').read_text()))"),
    code(
        "B = yaml.safe_load((P3 / 'baseline_summary.yaml').read_text())\n"
        "pl = B['plausibility']\n"
        "rows = {'specific thrust [N s/kg]': (pl['specific_thrust_N_s_kg']['model'], pl['specific_thrust_N_s_kg']['database_range_lt_250N']),\n"
        "        'TSFC [g/(kN s)]': (pl['TSFC_g_kN_s']['model'], pl['TSFC_g_kN_s']['db_min_median_max_n'][:3]),\n"
        "        'shaft speed [rpm]': (pl['shaft_speed_rpm']['model'], pl['shaft_speed_rpm']['db_min_median_max_n'][:3]),\n"
        "        'diameter [mm]': (pl['diameter_mm']['model_proxy'], pl['diameter_mm']['db_min_median_max_n'][:3]),\n"
        "        'length [mm]': (pl['length_mm']['model_proxy'], pl['length_mm']['db_min_median_max_n'][:3]),\n"
        "        'EGT [C]': (pl['EGT_C']['model'], pl['EGT_C']['db_min_median_max_n'][:3])}\n"
        "print(pl['comparison_set'])\n"
        "pd.DataFrame(rows, index=['model', 'database (min, median, max)']).T"
    ),
    md(
        "**Cross-check.** With the component efficiencies the coupled model produced, the 0-D cycle and "
        "pyCycle-CEA give the same thrust and TSFC:"
    ),
    code(
        "pd.DataFrame({k: v for k, v in B['cross_check_design_point'].items() if k in ('coupled', 'cycle_simple', 'pycycle_cea')})"
    ),
    md(
        "## 4. Operating line (sea level, speed-controlled)\n"
        "The ECU of a model turbojet sets the shaft speed; thrust, fuel flow and temperatures follow. "
        "Below about 30 % speed the nozzle has no pressure ratio left (labelled, not dropped)."
    ),
    code(
        "ol = pd.read_csv(P3 / 'baseline_operating_line.csv'); ok = ol[ol.converged]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))\n"
        "axes[0].plot(ok.n_frac * 100, ok.Fn_N, color=SERIES[0]); axes[0].set(xlabel='speed [% design]', ylabel='net thrust [N]')\n"
        "axes[1].plot(ok.n_frac * 100, ok.tit_K - 273.15, color=SERIES[1], label='TIT'); axes[1].plot(ok.n_frac * 100, ok.T05_K - 273.15, color=SERIES[2], label='EGT')\n"
        "axes[1].axhline(750, color=MUTED, ls=':', lw=1); axes[1].text(36, 755, 'JetCat max EGT 750 C', color=MUTED, fontsize=8)\n"
        "axes[1].set(xlabel='speed [% design]', ylabel='temperature [C]'); axes[1].legend(frameon=False)\n"
        "axes[2].plot(ok.W_air_kg_s, ok.pr_c, 'o-', color=SERIES[3], ms=3); axes[2].set(xlabel='air mass flow [kg/s]', ylabel='compressor PR')\n"
        "plt.tight_layout(); plt.show()\n"
        "ol.loc[~ol.converged, ['n_frac', 'failure_reason']]"
    ),
    md(
        "## 5. Materials: where TIT becomes a constraint\n"
        "The TIT bound is now 1000-1350 K. Feasibility comes from the turbine material "
        "(`configs/materials.yaml`): IN-713C (default) to 1253 K, MAR-M 247 (premium) to 1273 K, with zero "
        "gas-to-metal allowance. Above the default material: tagged *premium material required*; above all: "
        "infeasible but kept in the data."
    ),
    code(
        "from turboml.physics import mechanical as mech\n"
        "M = mech.MaterialSet.from_yaml()\n"
        "pd.DataFrame([{'TIT [K]': T, **{k: v for k, v in mech.turbine_temperature_check(T, M).items() if k in ('material', 'premium_material_required', 'feasible')}}\n"
        "              for T in (1100, 1200, 1250, 1260, 1270, 1280, 1350)])"
    ),
    md("## 6. Try it: one off-design point (a few seconds)"),
    code(
        "import time\n"
        "from turboml.physics.engine import Engine, design_engine, load_engine_config\n"
        "cfg = load_engine_config(); d = design_engine(cfg['baseline_design'], cfg); e = Engine(d, cfg)\n"
        "t0 = time.perf_counter(); p = e.off_design(0.9 * d.design_point.n_rpm, alt_m=1000.0, mach=0.3)\n"
        "print(f'{time.perf_counter() - t0:.1f} s:', {k: round(v, 3) if isinstance(v, float) else v for k, v in p.to_dict().items()\n"
        "      if k in ('converged', 'failure_reason', 'Fn_N', 'W_air_kg_s', 'pr_c', 'tit_K', 'T05_K', 'TSFC_g_kN_s')})"
    ),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[2] / "notebooks" / "03_coupled_engine.ipynb"
nbf.write(nb, out)
print("wrote", out)

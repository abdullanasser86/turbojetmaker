"""Build notebooks/00_environment_check.ipynb (run with nbconvert --execute afterwards)."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        "# Phase 0 — environment check\n\n"
        "This notebook checks that the three physics codes TurboML builds on run in our pinned "
        "environment, and shows what their outputs look like.\n\n"
        "1. **pyCycle**: NASA's cycle-analysis code, running its full-size turbojet example.\n"
        "2. **RadComp**: centrifugal-compressor meanline model, with a speed sweep in air.\n"
        "3. **TurboFlow**: axial-turbine meanline model, at one operating point.\n\n"
        "None of these are micro-turbojet results yet; they're plumbing checks. "
        "See `docs/LEARNING.md` for the vocabulary."
    ),
    code(
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "os.chdir(ROOT)\n"
        "print(subprocess.run([sys.executable, 'scripts/check_env.py'], capture_output=True, text=True).stdout)"
    ),
    md(
        "## 1. pyCycle `simple_turbojet` (unmodified upstream example)\n"
        "pyCycle reports in US units. Below we parse the summary line of each operating point "
        "and convert it to SI with `turboml.units`."
    ),
    code(
        "import pandas as pd\n"
        "from turboml import units as u\n"
        "proc = subprocess.run([sys.executable, str(ROOT / 'third_party/pycycle_examples/simple_turbojet.py')],\n"
        "                      cwd=ROOT / 'runs', capture_output=True, text=True)\n"
        "print('exit code', proc.returncode)\n"
        "lines = proc.stdout.splitlines()\n"
        "names = [l.split('POINT:')[1].strip() for l in lines if 'POINT:' in l]\n"
        "rows = [lines[i + 1].split() for i, l in enumerate(lines) if 'Mach      Alt' in l]\n"
        "df = pd.DataFrame([[float(x) for x in r] for r in rows], index=names,\n"
        "                  columns=['Mach', 'Alt_ft', 'W_lbm_s', 'Fn_lbf', 'Fg_lbf', 'Fram_lbf', 'OPR', 'TSFC_lbm_lbf_h'])\n"
        "si = pd.DataFrame({\n"
        "    'Mach': df.Mach, 'alt_m': df.Alt_ft * u.FT_TO_M,\n"
        "    'air_kg_s': df.W_lbm_s.map(u.lbm_s_to_kg_s), 'Fn_kN': df.Fn_lbf.map(u.lbf_to_N) / 1e3,\n"
        "    'OPR': df.OPR,\n"
        "    'TSFC_g_kN_s': df.TSFC_lbm_lbf_h.map(lambda t: u.tsfc_kg_per_N_s_to_g_per_kN_s(u.tsfc_lbm_per_lbf_h_to_kg_per_N_s(t))),\n"
        "})\n"
        "si.round(3)"
    ),
    md(
        "This is a **52 kN** engine with pressure ratio 13.5. Our target class is 0.05–1 kN "
        "with pressure ratio around 2–4. Phase 1 re-builds this model at micro scale."
    ),
    md(
        "## 2. RadComp: compressor speed sweep in air\n"
        "This uses the Schiffmann & Favrat (2010) impeller, a 20 mm-diameter heat-pump "
        "compressor (not a turbojet compressor), at fixed mass flow. Pressure ratio "
        "rises with tip speed, as Euler's turbine equation predicts."
    ),
    code(
        "import numpy as np, yaml, matplotlib.pyplot as plt\n"
        "from radcompressor.compressor import Compressor\n"
        "from radcompressor.condition import OperatingCondition\n"
        "from radcompressor.geometry import Geometry\n"
        "from radcompressor.thermo import CoolPropFluid\n"
        "db = yaml.safe_load((ROOT / 'third_party/radcomp_known_compressors.yml').read_text())\n"
        "geom = Geometry.from_dict(next(d for d in db if d['name'] == 'Schiffmann and Favrat')['geom'])\n"
        "in0 = CoolPropFluid('Air').thermo_prop('PT', u.ISA_SEA_LEVEL_P_PA, u.ISA_SEA_LEVEL_T_K)\n"
        "res = []\n"
        "for n_rpm in np.arange(140e3, 261e3, 10e3):\n"
        "    c = Compressor(geom, OperatingCondition(in0=in0, fld=in0.fld, m=0.010, n_rot=u.rpm_to_rad_s(n_rpm)))\n"
        "    ok = c.calculate()\n"
        "    res.append((n_rpm / 1e3, u.rpm_to_rad_s(n_rpm) * geom.r4, c.PR if ok else np.nan, c.eff if ok else np.nan, ok))\n"
        "rc = pd.DataFrame(res, columns=['N_krpm', 'U2_m_s', 'PR', 'eta_is', 'valid'])\n"
        "fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))\n"
        "ax[0].plot(rc.U2_m_s, rc.PR, 'o-'); ax[0].set(xlabel='tip speed U2 [m/s]', ylabel='total pressure ratio')\n"
        "ax[1].plot(rc.U2_m_s, rc.eta_is, 'o-'); ax[1].set(xlabel='tip speed U2 [m/s]', ylabel='isentropic efficiency')\n"
        "for a in ax: a.grid(alpha=.3)\n"
        "fig.suptitle('RadComp, Schiffmann & Favrat impeller, air, m = 0.010 kg/s (plumbing check)'); fig.tight_layout()\n"
        "rc.round(3)"
    ),
    md(
        "Invalid rows (NaN) are points where RadComp marks the operating point infeasible. "
        "Later phases record such failures as training data for the feasibility classifier "
        "instead of discarding them."
    ),
    md(
        "## 3. TurboFlow: single-stage axial turbine\n"
        "TurboFlow's own example, using the NASA cold-air turbine geometry of Kofskey et al. "
        "(1972). TurboFlow ships measured data for this turbine, so it's our candidate "
        "turbine validation case for Phase 2."
    ),
    code(
        "os.environ.setdefault('MPLBACKEND', 'Agg')\n"
        "import turboflow as tf\n"
        "cfg = tf.load_config(str(ROOT / 'third_party/turboflow_one_stage_config.yaml'), print_summary=False)\n"
        "solver = tf.compute_performance(cfg['operation_points'], cfg, export_results=False, stop_on_failure=True)[0]\n"
        "o = solver.problem.results['overall']\n"
        "pd.Series({k: float(o[k].iloc[0]) for k in ['efficiency_ts', 'efficiency_tt', 'mass_flow_rate', 'PR_ts', 'power']}).round(4)"
    ),
    md(
        "## Summary\n"
        "All three codes run in the pinned environment. The values above are also asserted as "
        "regression baselines in `tests/test_external_tools.py`. Next is Phase 1: the cycle "
        "model at micro-turbojet scale, plus an independent 0-D cross-check model."
    ),
]

nb = nbf.v4.new_notebook(
    cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}}
)
out = Path(__file__).resolve().parents[2] / "notebooks" / "00_environment_check.ipynb"
nbf.write(nb, out)
print("wrote", out)

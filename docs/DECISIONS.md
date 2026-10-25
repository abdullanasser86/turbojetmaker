# Decision log

Format: ID · date · decision · why · alternatives · status. Newest at the bottom.

### D-001 · 2026-10-24 · Python 3.11 + `uv` with a committed `uv.lock`
- **Why:** brief asks for 3.11. All tools support it (TurboFlow needs ≥3.11, BoTorch ≥3.11).
  Note: numpy ≥2.5 and scipy ≥1.18 require Python ≥3.12, so on 3.11 we get numpy 2.3.x /
  scipy 1.17.x — fine. `uv.lock` is cross-platform (one lock for the cloud container and the
  workstation).
- **Alternatives:** conda (heavier, slower solves; nothing here needs conda-only binaries).
- **Status:** accepted (Phase 0).

### D-002 · 2026-10-24 · Override TurboFlow's `numpy<2` and `CoolProp<7` pins
- **Problem:** TurboFlow 0.1.18 (latest) declares `numpy<2`, `CoolProp<7`, while OpenMDAO
  ≥3.42 and SALib ≥1.5.2 require `numpy>=2`. No single resolvable environment exists
  with the declared metadata.
- **Decision:** `[tool.uv] override-dependencies` lets TurboFlow install with numpy 2.3.5 and
  CoolProp 8.0.0.
- **Evidence:** TurboFlow's own regression suite (`tests/test_performance_analysis.py`, 7
  configs covering loss models Benner / Kacker–Okapuu / Moustapha / Ainley–Mathieson and all
  choking criteria) was run in this environment. All 188 reported mismatches are in
  solver bookkeeping only (`norm_residual`, `func_count`, `grad_count`); every physical
  output (efficiencies, torque, flow angles, enthalpies, …) agrees with the stored
  references to all 6 significant figures printed (the suite itself demands 10 digits).
- **Alternatives:** (a) a numpy-1.26 environment: forces OpenMDAO ≤3.41 *and* is incompatible
  with current pyCycle fixes; (b) separate venv for TurboFlow called via subprocess — kept
  as a fallback if a future TurboFlow/CoolProp combination breaks.
- **Status:** accepted; re-run TurboFlow's regression suite after any numpy/CoolProp upgrade.

### D-003 · 2026-10-24 · Install pyCycle from git commit `ee7e161` instead of PyPI 4.4.0
- **Problem:** PyPI `om-pycycle` 4.4.0 (Oct 2025) crashes in `thermo_add.py` ("setting an
  array element with a sequence") with numpy 2.x — reproduced with both OpenMDAO 3.45.1 and
  3.41.0 (pyCycle issue #116). The fix is on master (PR #117, May 2026) but unreleased.
- **Decision:** pin `om-pycycle @ git+…/pyCycle@ee7e16126a7a54b5fe35292c53b7eb4e9572620b`
  (reports version 4.4.1.dev0) with OpenMDAO 3.45.1.
- **Note:** the README compatibility table stops at pyCycle 4.2.0 ("OpenMDAO 3.10 or greater"),
  so it does not cover 4.4.x; compatibility was established empirically (example runs + our
  regression test).
- **Status:** accepted; switch to the next PyPI release once it contains PR #117.

### D-004 · 2026-10-24 · Constrain `numpy>=2.0,<2.4`
- **Why:** with numpy 2.4.6, pyCycle's `simple_turbojet.py` viewer fails ("only 0-dimensional
  arrays can be converted to Python scalars") — numpy 2.4 turned a long-deprecated
  conversion into an error, and pyCycle code still relies on it. numpy 2.3.5 works (with a
  DeprecationWarning). OpenMDAO also excludes 2.4.0.
- **Status:** accepted; revisit when pyCycle is numpy-2.4 clean.

### D-005 · 2026-10-24 · PyTorch from PyPI (`torch==2.14.1`, bundles CUDA 13.0 on Linux)
- **Why:** simplest reproducible source; on Linux the PyPI wheel already contains CUDA 13
  runtime libraries. (The PyTorch wheel index `download.pytorch.org` is also blocked by the
  cloud container's network policy.)
- **Implication:** the workstation's NVIDIA driver must support CUDA 13.0. On Windows the
  PyPI wheel is CPU-only — would need an explicit PyTorch index. **Open question to user.**
- **Status:** provisional until the workstation is confirmed.

### D-006 · 2026-10-24 · Config = YAML + pydantic, `null` for un-researched values
- Strict schemas (`extra="forbid"`) catch typos; `extends:` for study configs; `require()`
  makes un-researched values fail loudly instead of defaulting silently.

### D-007 · 2026-10-24 · Run logging with plain run directories (no MLflow yet)
- `runs/<UTC>_<name>/{config.yaml, meta.json, results…}` with git hash, dirty flag, versions,
  seed. MLflow can be added in Phase 5 if experiment comparison needs it.

### D-008 · 2026-10-24 · pre-commit with local hooks (`uv run ruff` / `uv run black`)
- Hooks use the lock-pinned tool versions, so pre-commit and CI/IDE never disagree.

### D-009 · 2026-10-24 · 0-D model uses pyCycle's NASA species data, its own equations
- **Why:** the cross-check must compare *cycle formulations* with matched assumptions. Sharing
  the published NASA 9-coefficient species data (read from `pycycle.thermo.cea.thermo_data.janaf`)
  means a disagreement points at a modelling bug, not at property data. Species values are
  independently tested against JANAF reference numbers (`tests/test_thermo.py`).
- **Differences that remain on purpose:** complete combustion (no dissociation) vs CEA
  equilibrium; consistent C atomic weight 12.0107 vs pyCycle's 12.017 (a typo, ≈0.05 % on
  fuel carbon content).

### D-010 · 2026-10-24 · Set pyCycle's fuel enthalpy from LHV and combustion efficiency
- **Finding:** pyCycle's `Combustor` takes the fuel enthalpy as an input (`mix_fuel…mix:h`) that
  defaults to 0 and is never set in the examples. With pyCycle's thermo data, h = 0 for C12H23
  corresponds to an LHV of 44.85 MJ/kg. Checked from the `simple_turbojet` burner energy
  balance: the implied fuel enthalpy is −0.08 MJ/kg ≈ 0. Real Jet A/A-1: ≥ 42.8 MJ/kg
  (ASTM D1655 minimum).
- **Decision:** `h_fuel = Σ h_f(products) / M_fuel + η_b · LHV` (see
  `turboml.gas.thermo.fuel_enthalpy_J_per_kg`). The same expression is used in both models.
  Combustion efficiency enters as the fraction of the LHV that is released.
- **Effect:** using the default would bias TSFC ≈ 4.6 % low (notebook 01, section 5).

### D-011 · 2026-10-24 · Altitude is geometric in TurboML; converted to geopotential for pyCycle
- pyCycle's US-1976 table is indexed by geopotential altitude (verified: 0.08 % pressure
  mismatch at 6 km without the conversion, < 0.01 % with it).

### D-012 · 2026-10-24 · pyCycle's nozzle `choked` flag is replaced by a pressure test
- `Mux.compute()` never assigns its `choked` output, so the flag is meaningless. A convergent
  nozzle is choked iff exit static pressure > ambient (relative tolerance 1e-4).

### D-013 · 2026-10-24 · Phase 1 pyCycle model is design-point only, with generic maps
- At the design point the compressor and turbine maps only set their own scaling, so the generic
  AXI5 / LPT2269 maps don't affect results. Off-design requires centrifugal-compressor and
  turbine maps from the Phase 2 meanline models. The wrapper documents this restriction.

### D-014 · 2026-10-24 · Failures are returned, not raised
- Both cycle models return `CycleResult(converged=False, failure_reason=...)` with labels
  `constraint_tit_below_t3`, `no_nozzle_pressure_ratio`, `no_net_thrust` and `no_convergence: …`.
  These become feasibility-classifier training data in Phase 4/5.

### D-015 · 2026-10-24 · OPEN: which cycle model generates training data (decision for user)
- See docs/PHASE1_REPORT.md: 0-D (5 ms) vs pyCycle CEA (≈1 s) vs pyCycle TABULAR (≈0.35 s,
  up to 3.6 % error).

### D-016 · 2026-10-24 · Training data from the 0-D model; pyCycle-CEA as cross-check (user decision)
- Supersedes the open question D-015. Data generation uses `cycle_simple` (≈5 ms/point).
- **Cross-check policy** (`turboml/physics/crosscheck.py`): pyCycle-CEA runs in parallel on a random 2 % of
  every generated dataset **and** on every final Pareto design. A point is flagged if |ΔFn/Fn| or |ΔTSFC/TSFC| > 1 %.
  If more than 5 % of cross-checked points are flagged → stop and report before training surrogates (`crosscheck.gate`).
- pyCycle-TABULAR dropped (removed from the wrapper).
- 0-D model refactored into component functions (`free_stream`, `intake`, `compressor`, `combustor`,
  `turbine_for_power`, `turbine_given_pr`, `convergent_nozzle`) sharing a `FlowState`, so Phase 3 meanline maps
  can drive them in an off-design matching solver.

### D-017 · 2026-10-24 · Phase 1 cross-check accepted at 63/64; metric unchanged (user decision)
- Primary metric stays |ΔFn/Fn|, |ΔTSFC/TSFC| < 1 %. Sample 8 is a documented known exception
  (`validation/phase1/known_exceptions.yaml`), pinned by a regression test.
- |ΔFn|/Fg is reported from now on as a **secondary, labelled** metric. It never decides pass/fail.

### D-018 · 2026-10-24 · Active bounds in `configs/bounds.yaml` (user decision)
- PR 2.0–5.0, air flow 0.08–1.9 kg/s, shaft speed 50–220 krpm, TIT 1000–1250 K (provisional),
  impeller tip speed ≤ 500 m/s (configurable constraint). Each bound has its source.
- Checked against the 109-engine review database (`scripts/data/derive_bounds.py`); see the Phase 2 report
  for proposed updates (TIT upper limit).

### D-019 · 2026-10-24 · Source documents are not committed
- Uploaded PDFs are copyrighted. Committed: extracted tables (CSV with per-row citation), the parsers
  (`scripts/data/`), and digitized data clearly labelled as digitized with reading uncertainty.

### D-020 · 2026-10-24 · Device-agnostic training (`turboml.device.get_device`)
- CUDA if available, else CPU; override with `TURBOML_DEVICE`. `scripts/check_env.py` verifies installed
  versions against `uv.lock` and smoke-tests torch/GPyTorch/BoTorch on the selected device.

### D-021 · 2026-10-24 · Own centrifugal-compressor meanline model; TurboFlow and RadComp as cross-checks
- **Findings in TurboFlow 0.1.18's centrifugal-compressor module** (its axial-turbine module is unaffected):
  1. Wiesner slip evaluates `np.cos()` of the trailing-edge angle **in radians although the angle is in degrees**
     (wrong slip factor; NaN for |β2b| ≳ 33°).
  2. The `algebraic` vaneless diffuser with the Oh loss set is **lossless**. Oh's vaneless loss (Stanitz) is a
     thermodynamic definition of loss, so the loss residual is satisfied identically; entropy and p0 are unchanged
     for any skin-friction factor (verified: Cf 0.02 vs 0.5 gives identical p0). Reported stage efficiency is
     effectively impeller-only.
  3. The `1D` vaneless diffuser model raises `TypeError: Expected numeric dtype, got object` in our pinned stack.
- **Decision:** write `turboml/physics/compressor.py`: impeller losses = Oh, Yoon & Chung (1997) set as implemented
  in TurboFlow's `loss_model_oh.py` (MIT, ported with attribution); 1-D vaneless diffuser with wall friction
  (real p0 loss); selectable slip with Δσ uncertainty; explicit choke and surge criteria; our own thermo
  (no CoolProp). Fast enough for Phase 3/4.
- TurboFlow's impeller (with our slip-bug fix, `compressor_turboflow.py`) serves as a port-verification
  cross-check. RadComp (Schiffmann & Favrat loss set) serves as an independent-loss-set cross-check.

### D-022 · 2026-10-24 · Incidence loss relative to the blade (default) instead of TurboFlow's reading
- TurboFlow reads Conrad's incidence loss as ½·f·W_t1² (the whole inlet tangential relative velocity), which doesn't
  depend on mass flow. Speed lines then have no off-design incidence penalty and no PR peak.
- Default `incidence_model="blade_relative"`: ½·f·(W1·sin(|β1| − β1b))² at the mean radius. The TurboFlow reading
  stays available (used for the port-verification test).

### D-023 · 2026-10-24 · Compressor surge criterion (preliminary design, NOT validated)
- Surge point = highest flow where either an interior PR maximum occurs or the vaneless-diffuser critical flow angle
  (Senoo & Kinoshita 1977, via RadComp's fit) is exceeded. The impeller diffusion ratio W1t/W2 is output but not used
  (no sourced threshold yet). The available KJ-66 data contains no surge points.

### D-024 · 2026-10-24 · KJ-66 compressor calibration method
- Blind run first (reported). Then 2 effective parameters (Δσ, diffuser Cf) fitted on ONE measured speed (90 krpm),
  validated on the other four. Values in `configs/components/kj66_compressor_calibration.yaml`. Transfer to other
  designs is an open decision for the user.

### D-025 · 2026-10-24 · Axial turbine: TurboFlow, run on air
- TurboFlow's axial-turbine module reproduces Kofskey (1972) measured data untuned (η_ts ≈1.4 pts, ṁ ≈1.1 %).
- Air instead of combustion products: work −0.2…−0.5 %, choked flow function +0.3…+0.5 % (`scripts/phase2/turbine_gas_error.py`).
- It takes ~3 s per point, too slow for Phase 3/4 matching at scale. Options are in the Phase 2 report.

### D-026 · 2026-10-25 · torch 2.10.0 (CUDA 12.8 build) for the RTX 4080 Laptop GPU
- User requirement: PyTorch built for CUDA 12.x with GPyTorch/BoTorch matched in the lock; Ada GPU (sm_89, 12 GB).
- `torch==2.10.0` is the newest release whose default PyPI Linux x86-64 wheel is built against CUDA 12.8 (`2.10.0+cu128`,
  cuDNN 9.10). It contains sm_89 kernels. It needs an NVIDIA driver that supports CUDA 12.8 (R570 or newer). Locked with it:
  gpytorch 1.15.2, botorch 0.18.1, linear-operator 0.6.1.
- **Windows caveat:** PyPI ships only CPU torch wheels for Windows. A CUDA build there needs the PyTorch package index
  (`https://download.pytorch.org/whl/cu128`) as a uv index for torch. That index is blocked from the build container, so the
  lock can't be resolved for it here. The user's OS is an open question (Phase 3 report).
- Precision policy: float64 for GPs, float32 for neural networks. Exact GPs up to ~10k points per output; sparse/variational GPs
  or deep ensembles above that. KeOps only if profiling shows a need.

### D-027 · 2026-10-25 · CPU worker count and resumable checkpoints
- `compute.n_workers` (configs/default.yaml): `null` → physical cores − 1 (at least 1). It can be overridden with
  `TURBOML_N_WORKERS` (`turboml/compute.py`).
- `turboml/checkpoint.Checkpointer`: atomic save (temp file + rename), keep the last N, store and restore the RNG states.

### D-028 · 2026-10-25 · Own fast axial turbine meanline model (user decision 2026-10-25, item 2)
- `turboml/physics/turbine_meanline.py` uses TurboFlow 0.1.18's equations (Benner losses ported verbatim, Aungier
  deviation, critical-Mach choking, interspace model) with a **marching** solver. The flow is solved plane by plane,
  with one scalar unknown at a time (mass flow; or stator-exit pressure when the stator is choked). The thermo is our
  NASA-polynomial model (combustion products supported).
- Solver: fused Newton/secant row iteration, choke structure determined first (stator/rotor critical flows), regime
  decided from the choked-branch onset pressure, warm start between calls. A safety net rejects any point whose exit
  pressure misses the target (> 1e-7 p0).
- Speed: median 7.5 ms per point in sweeps/engine iterations, ~17 ms cold (target < 10 ms per point: met in sweeps,
  not for an isolated cold point).
- TurboFlow stays available (`turboml/physics/turbine.py`, `AxialTurbine.from_geometry`) as the higher-fidelity
  cross-check and a possible multi-fidelity level. Interface `TurbineModel` (evaluate/with_gas) is generic, so a radial
  turbine can be added later.
- Gap between the subsonic and choked branches (< ~0.3 % mass flow, inherent to the critical-Mach criterion): the
  choked branch is used there.

### D-029 · 2026-10-25 · Benner incidence sign: keep TurboFlow's effective behaviour (OPEN)
- TurboFlow 0.1.18 `get_incidence_loss` sets `cascade_type = ["cascade_type"]` (a list literal), so its intended
  rotor sign (+1) is never applied and every row uses −1.
- Default `incidence_sign="turboflow"` (−1 everywhere), because it reproduces TurboFlow's validated results and fits
  the Kofskey measurements better. Mean η_ts error is −1.2 pts with this sign and −2.0 pts with the intended sign
  (`rotor_plus`); at 50 % speed (+27° rotor incidence) the errors are +0.1 and −2.6 pts.
- Open until the Benner et al. (1997) paper is checked for its angle convention. Report the bug upstream either way.

### D-030 · 2026-10-25 · Turbine sizing rules (design mode)
- Velocity triangles from φ, ψ, R (constant Cx and radius, repeating-stage relations). Zero-incidence leading edges.
  Gauging angles inverted from the Aungier deviation at the design Mach number (Ma_crit first guess 0.95). Pitch from
  Zweifel 0.8, chord from aspect ratio, t_max from TurboFlow's camber rule. TE, LE and clearance come from ratios in
  `configs/components/axial_turbine_design.yaml`.
- `design_stage` scales the annulus (blade heights) and the PR with a 2×2 Newton, so the meanline model passes exactly
  the design flow at exactly the design work.

### D-031 · 2026-10-25 · Compressor sizing for new designs (KJ-66-like family)
- Inducer tip radius from the KJ-66 tip flow coefficient C1/U1t; hub/tip ratio from the KJ-66; blade inlet angle with
  the KJ-66 incidence; r2 solved for the target PR; b2 from the KJ-66 exit flow coefficient Cm2/U2; lengths and
  diffuser radius scaled with r2; blade count, backsweep and throat ratio from the KJ-66. The reference point is the
  calibrated model's peak-η point at 110 krpm.
- This keeps new compressors in the regime where the calibration was made, and it leaves shaft speed as an
  independent design variable.

### D-032 · 2026-10-25 · Engine matching formulation
- Speed-controlled off-design (the ECU sets N). Unknowns: air flow, TIT, turbine PR_ts. Residuals: turbine flow, shaft
  power, nozzle flow.
- Solver: damped Newton with an FD Jacobian plus Broyden updates and a line search; failed evaluations count as a
  failed step. Continuation in 10 % speed steps, and in the inlet state for altitude and flight Mach.
- Nozzle inlet total pressure = turbine-exit absolute total pressure (exit swirl recovered; labelled assumption).
- Off-design pressure-loss fractions (intake, burner, nozzle) scale with (corrected flow)² relative to the design
  point (dynamic-head scaling). With constant fractions, TIT rose at part speed (1602 K at 50 % speed), the opposite of
  real engines.

### D-033 · 2026-10-25 · Materials and the 1350 K TIT bound (user decision 2026-10-25, item 3)
- TIT bound raised to 1350 K; feasibility from `configs/materials.yaml`. Metal temperature = TIT − allowance
  (allowance 0 K, conservative). Above the default material (IN-713C, 1253 K) → "premium material required" (MAR-M
  247, 1273 K); above all → infeasible, labelled, kept.
- Blade root stress: σ = K ρ ω² A / 2π. Status "unchecked: material_data_missing" until allowable stresses are sourced.

### D-034 · 2026-10-25 · Envelope and mass proxies
- D = 1.627 × impeller diameter (KJ-66 casing ratio); L = 2.59 D (median of 87 database engines); m = k D² L with
  k = 432 ± 43 kg/m³ (14 JetCat PRO variants). `scripts/phase3/fit_envelope.py` reproduces these.

### D-035 · 2026-10-25 · KJ-66 calibration as provisional defaults with an uncertainty band (user decision, item 1)
- Band = 2σ from the least-squares Jacobian on the 10 calibration rows (90 krpm only): Δσ [−0.119, −0.089],
  Cf [0.086, 0.102]. It is statistical only (no model-form error). Phase 4 samples a fraction (proposal 20 %) of the
  designs uniformly inside it. Validation errors are only ever reported on the held-out speeds.

### D-036 · 2026-10-25 · Incidence sign: both selectable, TurboFlow's effective sign stays default (user decision)
- `configs/engine_baseline.yaml` → `turbine_design.incidence_sign`: `turboflow` (default) or `rotor_plus`. Both
  validation runs are recorded:
  - Kofskey vs measured: η_ts mean −1.20 / −2.00 pts, torque −0.53 / −2.38 % (turboflow / rotor_plus).
  - Kofskey vs TurboFlow: turboflow sign within 0.1 % / 0.06 pts; rotor_plus up to 7.5 % power.
  - Sobol 512 points: rotor_plus − turboflow η_ts mean +0.04 pts, but single points jump to another solution branch
    (−4.3 … +47 pts).
- Reasoning: the released behaviour fits the only measured data; the paper (Benner et al. 1997) is paywalled, so the
  sign stays an open modelling uncertainty. Issue text for TurboFlow drafted in `docs/upstream_issues.md`; posting is
  the user's call.

### D-037 · 2026-10-25 · Turbine creep checks from NiDI No. 393 (user decision, materials item)
- Data: Tables 9/10 (100 h / 1000 h rupture), Table 1 (density) for Alloy 713C (default) and MAR-M 247 (premium,
  test bars machined from blades) → `validation/cases/ni393_cast_rupture.csv`. One printed typo (713C 1000 h at
  1400 °F: 458 MPa vs 65 ksi = 448) is resolved by using the ksi column throughout.
- Method: interpolation of log-stress in temperature at a tabulated life (and in log-life between 100 and 1000 h).
  Larson–Miller (C = 20, quadratic fit, max error 10 % for 713C) only for other lives. No extrapolation above the
  tabulated temperatures (→ "TIT_*_above_material_data"); strength held constant below them.
- Settings (preliminary-design choices): life 100 h, safety factor 1.5 on stress, taper factor 1.0.
- Metal temperatures: rotor = rotor-inlet relative total temperature (meanline); NGV = TIT + PF (TIT − T3).
- Stresses: rotor = centrifugal (AN²) + gas bending; NGV = gas bending. Cantilever with a uniform load;
  Zs = (8/105) c t² is derived for a parabolic profile and ignores camber (conservative).
- Pattern factor: no verifiable cited range found (web search 2026-10-25), so `combustor_pattern_factor: null`. The
  dataset stores PF_max, the largest PF the NGVs survive, and feasibility can be recomputed once PF is sourced. A design
  that fails at PF = 0 is infeasible.
- Infeasible only if the selected material fails. Active constraints reported: AN2, creep_rotor, creep_ngv,
  TIT_rotor/ngv_above_material_data, tip_speed.
- Consequence: the Phase 3 baseline (557 MPa centrifugal at a 1088 K rotor) is infeasible (AN2) with K = 1 and SF 1.5.

### D-038 · 2026-10-25 · Calibration band inflated for model-form error (user decision)
- Common factor k on both 2σ half-widths, increased until ≥ 95 % of the 34 held-out measured points lie inside the
  model's prediction envelope (3 × 3 grid). Result: k = 2.75 (34/34; k = 2.5 → 32/34). Inflated band:
  Δσ [−0.145, −0.063], Cf [0.072, 0.115]. 20 % of the Phase 4 designs sample it uniformly.
- Caveat: the held-out points now size the band, so they are no longer independent evidence for its width.

### D-039 · 2026-10-25 · DePaola values and off-design loss scaling confirmed (user decision)
- "Nozzle ≈ 0.95" is the nozzle total-pressure ratio Pt9/Pt7 (Mattingly convention), with P0/P9 ≈ 1. The inlet 0.98
  is also a total-pressure ratio. The implementation already used `dp_nozzle_frac = 0.05`. To be re-checked against
  the thesis text when it arrives.
- Loss scaling with (corrected flow)² is accepted. Source: Walsh & Fletcher, *Gas Turbine Performance* (2nd ed.,
  Blackwell 2004), component pressure-loss scaling with corrected flow squared (dynamic head). Chapter-level citation;
  page to be checked.

### D-040 · 2026-10-25 · Compressor speedups (profiling pass)
- 90 % of a compressor evaluation was the vaneless-diffuser march: each RK4 stage solved continuity with a bracketed
  search (~21 thermo calls). It is replaced by a Newton step warm-started from the previous step,
  d(ρCmA)/dCm = ρA(1 − Cm²/a²), with the bracketed solver kept as a fallback (and for choke detection).
- Results are identical (max difference 7e-14 over 40 test points, same failure labels). The compressor is ~4×
  faster. On the baseline engine (`scripts/phase4/profile_engine.py` → `validation/phase4/profiling.yaml`, warm
  cache): design 0.90 s instead of 2.97 s, the three Phase 4 off-design points 0.71–0.95 s instead of 2.6–4.1 s
  (3.3–4.3×), outputs equal to 3e-11. No numba, no rewrite.
- The next hot spot is the gas-property inversion `thermo.T_from_h` (~39 % of the time after the speedup). It is left
  alone: the full 2000-design run already fits in minutes to tens of minutes.

### D-041 · 2026-10-25 · Robust compressor sizing and adaptive engine continuation
- Compressor sizing: r2 is bracketed (steps of ×1.12, bisection toward the edge of the working region) and then found
  by brentq. The exit width is made consistent with the exit density by a secant iteration, with a lower density
  guess after a choked first try. Before this, failures depended on the iteration path; now a design fails only if no
  working radius reaches the PR (labels design_compressor_pr_not_reached_<choke>, r2_below_inducer, ...).
- Off-design continuation is adaptive in speed and inlet state: a failed step is halved (down to 1/64) and the initial
  guess is speed-scaled (m ∝ N, PR_t − 1 ∝ N²). Before this, 80 % / idle points failed only because of the starting
  guess.

### D-042 · 2026-10-25 · Phase 4 dataset design (scope approved by the user)
- 2000 designs, scrambled Sobol (first 2000 of 2^11), 7 variables in the configs/bounds.yaml box. Mass flow is sampled
  on a log scale; the other variables linearly. 4 points per design with per-point failure reasons.
- 15 % test split frozen before evaluation (`data/splits/phase4_v1_split.csv`).
- Append-only resumable JSONL. Workers = physical cores − 1.
- pyCycle-CEA cross-check on a random 2 % of designed engines.

### D-043 · 2026-10-25 · Pilot dataset copied into validation/phase4 (small, for the notebook)
- `datasets/` stays git-ignored (regenerate from seed + config). The 256-design pilot's flattened CSVs (~0.4 MB) are
  committed as `validation/phase4/pilot_phase4_v1_*.csv`, so notebook 04 and the report numbers can be checked
  without re-running. The pilot was regenerated at commit e2f2a36 and was bit-identical to the first run.
- The full run waits for the sampling-parametrization decision (Phase 4 report, decision 1).

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

### D-044 · 2026-10-25 · Reference data from the user's extracts, tagged by data type (user decision)
- The PDFs could not be uploaded; the user extracted the numbers. They are in `validation/cases/` with citations:
  - `kj66_engine_specs.csv`: Xiang 2017 / thesis Table A-i.1 and Murray 2009 Table 1.2;
  - `microturbine_reference_assumptions.csv`: DePaola 2020 and Murray 2009 assumptions, DePaola Table 2 ranges and
    eq. 58;
  - `jetcat_specs_user_2026.csv`;
  - `combustor_pattern_factor.csv`.
- Every value carries MEAS / CFD / MFR / ASSUM. Weighting MEAS > CFD > MFR > ASSUM. The error floor against MFR
  data is 10 % (same-model spread, DePaola, MEAS).
- The component values in `configs/engine_baseline.yaml` are unchanged and now carry data types:
  - DePaola's η_b 0.95, η_m 0.99 and the inlet/nozzle ratios are kept; Murray's 0.90 / 0.995 are recorded but not
    used.
  - The burner Δp of 10 % is tagged CFD (Gieras & Stańkowski via DePaola).
- The KJ-66 builds of Xiang (75 N, 117 krpm, PR 2.2) and Murray (92.3 N, 128 krpm, PR ~2.54) are treated as a spread,
  not a contradiction.

### D-045 · 2026-10-25 · Pattern factor, blade taper as At/Ah, MAR-M 247 no-data status, DePaola sanity flags
- Pattern factor PF = 0.25 for feasibility (never lower), sensitivity 0.15–0.35. Sources: Fear 1972 (MEAS < 0.24) and
  US 5,265,425 (0.25–0.30). PF_max is still stored.
- Taper: DePaola eq. 58 (Mattingly), AN² = 3600 σ / (π ρ (1 + At/Ah)), is our σc = K ρ ω² A / 2π with
  K = (1 + At/Ah)/2. Default At/Ah = 1 (K = 1). Sensitivity At/Ah = 0.34 (K = 0.67).
- MAR-M 247 gap: if IN-713C fails and MAR-M 247's rupture data don't reach its metal temperature (> 1311 K), and nothing
  the data cover fails, the status is `unchecked_no_material_data` → infeasible. No extrapolation.
- DePaola Table 2 (AN² ≤ 5e10 in² rpm² = 3.2258e7 m² rpm², ψ 1.4–2.0, exit Mach 0.4–0.5, exit swirl 0–40°) are
  SANITY FLAGS on every design. They don't make a design infeasible.
- Effect on the Phase 3 baseline: the NGV metal temperature is now 1330 K. IN-713C passes there (data to 1366 K);
  MAR-M 247 has no data. The baseline remains infeasible on AN².

### D-046 · 2026-10-25 · Option C implemented as approved: tip speed and turbine hub/tip prescribed (dataset phase4_v2)
- Compressor (`engine.size_compressor_tip_speed`):
  - At fixed (W, PR) the sized wheel's tip speed falls as N rises (higher specific speed, better efficiency), until r2
    would drop below the inducer. N is found so that U2(N) = U2_target.
  - Labels: `design_compressor_tip_speed_too_low_for_pr` / `_too_high_for_pr`. A warm start of r2 was tried; it
    changed one result (path dependence near choke), so it was reverted.
- Turbine (`turbine_design.design_stage_hub_tip`): φ is found by the fixed point φ ← φ·h/h_target on the converged
  stage. An axial velocity at or above the exit stagnation sound speed is rejected with a label.
- Bounds:
  - U2 350–500 m/s. The upper limit is the user's; the lower limit is DERIVED from a sizing scan (PR 2.0 needs
    ≥ ~355 m/s in this family).
  - Rotor hub/tip 0.5–0.95 (TurboFlow range).
- Dataset `phase4_v2`: seed 20261026, own frozen split, pilot 200 with a stop rule at 10 % feasible. v1 is kept
  unchanged.

### D-047 · 2026-10-25 · KJ-66 compressor "rebuild": nothing to remove (same geometry already used)
- The user's Xiang 2017 values (r2 33 mm, hub 6.25 mm, shroud 22.5 mm, 10 blades) are exactly the geometry used since
  Phase 2. It came from the online-first version of the same paper, doi:10.1177/0954410016644632.
- No calibration factor is removed:
  - Δσ stands in for the unpublished blade exit angle / backsweep;
  - Cf stands in for the 15-vane wedge diffuser and 90° turn, whose vane geometry is unpublished.
- Recalibration on the 90 krpm measured points with the current code reproduces Δσ −0.1041 and Cf 0.0937
  (`scripts/phase4/kj66_rebuild_check.py`).
- The Xiang CFD efficiency drop (0.73 → 0.55, 80 → 117 krpm) is not reproduced. The model stays at 0.76–0.77, in line
  with the MEASURED 0.77 → 0.75 (70–110 krpm). Under the MEAS > CFD weighting the model is kept.

### D-048 · 2026-10-25 · Idle-EGT investigation: diagnostic hook only, no model change
- `Engine.diag_eta_delta` (diagnostic, default None) shifts a component efficiency at unchanged PR and flow
  capacity. `EnginePoint` gains `T05_static_K` and `turbine_exit_mach` for the thermocouple reading. With the hook
  off, results are bit-identical (test).
- Findings: `docs/VALIDATION.md` and the Phase 4 report. None of the four suspects explains the idle EGT. Nothing was
  tuned.

### D-049 · 2026-10-25 · Pilot v2 stopped by the stop rule (5.5 % feasible < 10 %)
- 185 of 200 designs fail in design mode: 161 `tip_speed_too_low_for_pr`, 20 turbine, 4 `too_high_for_pr`.
- 11 of the 15 designed engines are feasible. Relaxing all creep settings together gives 13, so the conservative
  settings are NOT the cause.
- Cause: in this KJ-66-like compressor family, PR and U2 are tied (no design above PR ~3.2 can be built at
  ≤ 500 m/s), and the reachable U2 band at a given PR is only ~50 m/s wide. Independent sampling of PR and U2 misses it.
- The full run was not started; the next step is the user's decision (Phase 4 report).

### D-050 · 2026-10-25 · Windows 11 workstation: CUDA torch from the PyTorch index, UTF-8, spawn-safe code
- The user's OS is Windows 11. PyPI's Windows torch wheel is CPU-only, so `pyproject.toml` adds the PyTorch cu128
  index (`explicit`, used only for torch on `sys_platform == 'win32'`). `uv.lock` holds torch 2.10.0+cu128 on Windows
  and 2.10.0 from PyPI elsewhere; both are CUDA 12.8 builds.
- `scripts/check_env.py`:
  - picks the platform's locked version;
  - fails on Windows if torch is not a +cu12x build, does not see the GPU, or Python's UTF-8 mode is off;
  - has no POSIX-only module imports (`resource` → psutil).
- Text I/O is explicitly UTF-8 in `turboml/`, `scripts/` and `tests/` (69 calls); Windows' default code page is
  cp1252. The README sets `PYTHONUTF8=1` for the notebooks.
- Parallel work uses joblib/loky (spawn-like) with top-level worker functions. Every script runs only under
  `if __name__ == "__main__"`; the notebook builders were the last scripts without a guard.
- `scripts/phase4/windows_smoke_test.py` checks that serial, a "spawn" pool and loky give identical records, and that
  the real generator runs with 2 workers and resumes. It passes in the Linux container; the user should run it on the
  laptop.

### D-051 · 2026-10-25 · Murray (2009) data: transcribed tables, digitized run plots, and what the numbers are
- Upload: thesis pp. 121–171 (Ch. 7, references, Appendices B–J). **Figs 5.1–5.3 and the maps 5.12–5.19 were not in
  it.**
- Appendix D (15 tables, vaneless EXHAUST-diffuser run 3, 48–120 krpm) was TRANSCRIBED to
  `validation/cases/murray2009/murray2009_appD_running_data.csv`, with Murray's printed uncertainties.
  Consistency checks pass: PR = P03/P02 to 0.03 %; FAR and corrected flow within 0.6 %.
- Table H.1 (pp. 157–159) shows what the numbers are:
  - T02 and T05 are thermocouple readings corrected to total temperature with r = 0.68.
  - **T04 is calculated** from T05 and the compressor work / η_m (0.995), even though it is printed in bold.
  - So Murray's turbine efficiency is derived, not measured.
- The Appendix C run plots (incl. the three NOZZLE = turbojet runs, down to 28 krpm) were DIGITIZED
  (`scripts/data/digitize_murray_runplots.py`) at steady plateaus: T5 reading, P03 gauge, bellmouth head P01 − P1.
  - The digitizer was checked against Appendix D on Fig C.7, which is the same run: rms 5.8 °C, 0.41 kPa,
    0.004 kPa. The stated uncertainty is 2 × rms.
  - The plots' axis label "T5 × 0.1 (K)" is wrong: the values match Appendix D only in °C.
  - Mass flow for the nozzle runs is derived from the bellmouth head with C = m/√(ρ ΔP) = 0.01041 m², inferred from
    Appendix D.

### D-052 · 2026-10-25 · Murray as the second KJ-66 validation case: calibration and validation kept apart
- Compressor: the stored calibration (fitted on the 90 krpm line) is not changed. Points within 3 krpm of 90 krpm
  (D.6, D.12) are reported as near-calibration; the other 13 as validation.
- Turbine: the KJ-66 turbine geometry is not available, so the check is indicative only (an assumed replica family,
  ψ 1.0/1.5/2.0, sized at the 120 krpm point).
- Engine (turbojet runs): replicas use Murray's ambient (98.9 kPa, 294.7 K) and the measured W and PR at 120 krpm.
  ONE parameter (TIT, i.e. the nozzle area) is fitted to the measured 120 krpm EGT reading. Every other speed
  (28–112 krpm) is a prediction.

### D-053 · 2026-10-25 · Option A: tip speed and tip-speed specific speed sampled, PR is an output (user decision)
- n_s,U = ω √Q01 / U2^1.5 (`engine.tip_speed_specific_speed`). It equals the usual specific speed up to the factor
  ψ_is^0.75, but needs no PR.
- `engine.size_compressor_direct` sizes the wheel directly from (W, U2, n_s,U), with no search: 0.04–0.07 s instead
  of 3–11 s.
- Bounds: U2 350–500 m/s; n_s,U 0.35–0.90. Both are derived from a sizing scan: every wheel is buildable up to 0.95,
  efficiency peaks near 0.9, and is below ~0.6 under 0.35. The KJ-66 itself is at 0.68. Every sample in the box is a
  buildable wheel (test).
- **PR range.** The user's wording was "roughly 1.9–3.3". With these bounds PR lands at 1.7–3.35 (scan) and 1.69–3.19
  in the v3 pilot. That brackets the real KJ-66/JetCat class (KJ-66 PR 2.2–2.54, JetCat P90 ≈ 2.35).
- **OPEN ITEM:** this excludes higher-PR micro engines (up to ~5 per the micro-turbojet review, Anjomrouz et al.
  2023). Reaching them needs a wider compressor family (option D: backsweep / work coefficient as a variable), which
  waits on blade-angle data.
- Dataset `phase4_v3`: seed 20261027, own frozen 15 % split, turbine rotor-exit hub/tip prescribed (as in v2). v1 and
  v2 are kept.

### D-054 · 2026-10-25 · Pilot v3 stopped by the stop rule (8.5 % feasible < 10 %)
- Compressor: no failures (option A works).
- Turbine: 149/200 fail in sizing. Above hub/tip 0.8 almost every design fails (the axial velocity would be
  supersonic); 0.7–0.8 fails 82 %; 0.5–0.7 about 50 % (limit loading / no convergence, worse at high ψ).
- Built engines: 17/51 feasible. Active constraints: NGV metal above the rupture data (31: with PF 0.25 every design
  above TIT ≈ 1145 K; IN-713C data end at 1366 K) and AN² (29).
- Relaxing all creep settings gives 16.5 %; a single setting gives at most 10.5 %.
- The full run was not started.

### D-055 · 2026-10-25 · Option T1: turbine hub/tip sampled inside its buildable range (user decision)
- The sampled variable is the position f ∈ [0, 1] in the per-design range [0.5, min(0.95, λ_M)], i.e.
  hub/tip = 0.5 + f (λ_hi − 0.5). 0.5 and 0.95 are TurboFlow's recommended limits. λ_M is the hub/tip at which the
  rotor-exit absolute Mach number reaches 0.5, the upper end of DePaola 2020 Table 2 (exit Mach 0.4–0.5).
- Derivation (`turbine_design.max_hub_tip_for_exit_mach`, computed per design before sizing, with the same first-guess
  states as `size_axial_stage`):
  1. U = √(Δh0/ψ), r_m = U/ω, exit swirl C_θ3 = U(1 − R − ψ/2).
  2. h03 = h04 − Δh0; p03 comes from the first-guess stage efficiency.
  3. At the limit, C3 = 0.5 a3 with T3 = T(h03 − C3²/2); C_x = √(C3² − C_θ3²).
  4. Continuity: A = m/(ρ3 C_x) = 2π r_m h.
  5. λ = (r_m − h/2)/(r_m + h/2).

  A higher hub/tip gives a smaller annulus, a faster flow and a higher exit Mach number, so λ_M is the upper end of
  the range.
- If even hub/tip 0.5 exceeds Mach 0.5 (high ψ: small mean radius), the range is empty. The design fails with
  `design_turbine_no_buildable_hub_tip` (labelled, kept).
- Check: at f = 1 the converged stage's exit Mach number is 0.49 (baseline; test ±0.03), because the first-guess
  density differs slightly from the converged one.
- DePaola's LOWER exit-Mach bound (0.4) is not imposed; it stays a sanity flag (user instruction: upper limit only).

### D-056 · 2026-10-25 · High-TIT designs limited to ~15 % of each dataset (user decision)
- TIT stays sampled over 1000–1350 K, so the feasibility classifier learns the material boundary. The Sobol coordinate
  is mapped piecewise-linearly: u ∈ [0, 0.85] → 1000–1150 K, u ∈ [0.85, 1] → 1150–1350 K. The design gets exactly
  15.0 % above 1150 K, in the pilot and in the full set.
- Reporting: feasibility inside and outside the band, and every infeasible high-TIT design with its active constraint
  and material status (`pilot_feasibility_analysis.py`, dataset card).
- **OPEN ITEM: higher-temperature rupture data needed.** With PF 0.25 the hottest NGV exceeds IN-713C's data
  (1366 K) above TIT ≈ 1150–1190 K, and MAR-M 247's data stop at 1311 K. Data for MAR-M 247 above 1900 °F, or another
  cast alloy, would open this range.

### D-057 · 2026-10-25 · Murray Chapter 5 digitized (Figs 5.1–5.3, 5.12–5.19)
- Scatter maps: marker centroids from pixel components (`scripts/data/digitize_murray_ch5.py`).
  - Axes are calibrated from the x-tick marks and the gridlines (40-krpm marker check: 0.0576 / 1.1305 vs 0.058 / 1.13
    read by eye).
  - Overlapping markers merge into one cluster centroid (n_markers_est).
  - Speed class: Murray interpolated the points to constant corrected speed (p. 87). Compressor classes are
    separated by PR band; efficiency points are matched to the PR map by corrected flow; turbine classes by PR band.
  - Uncertainty: u_dig = half the marker/cluster size + 1 px. u_pub = Murray's propagated uncertainty, from Appendix D
    at the nearest corrected speed. It matches the plotted bands of Figs 5.13/5.15/5.17/5.19 in spot checks (e.g.
    compressor η ±0.10 at 50 krpm, turbine η ±0.25 at 32 krpm).
- Line plots:
  - Figs 5.1/5.2: the 7 runs overlap within the line width, so the all-run envelope (min/max) is stored.
  - Fig 5.3: vertices were read by eye and snapped to the curve pixels. Check against Appendix D (vaneless run 3 =
    same run): rms 1.2 K, max 2.5 K. The stated uncertainty is 5 K.
  - The line charts have no right frame and no x ticks, so x is calibrated from the y axis and the end of the
    gridlines. The legend can cut the top gridline, so the longest gridline run is used.

### D-058 · 2026-10-25 · Measured KJ-66 maps vs the compressor calibration: contradiction at 110–120 krpm, NOT refitted
- The stored calibration (Δσ, Cf fitted on the measured 90 krpm line) is unchanged (user instruction: tell before
  recalibrating).
- **50–100 krpm** (validation): PR within −1.2 … +2.0 % (within Murray's published uncertainty), flow offset −3 … +0.6
  %, efficiency ±3 pts rms.
- **110 and 120 krpm**: the model over-predicts PR by +4.1 % and +7.1 %, beyond the published uncertainty on every
  point, and efficiency by +2.3 / +6.1 pts. 120 krpm (U2 ≈ 415 m/s) lies beyond the 70–110 krpm data the calibration
  was built on. **This contradicts the calibration at high tip speed.**
- Consequence for the datasets: designs with U2 above ~380–415 m/s (sampled up to 500 m/s) use a compressor model
  that is optimistic in PR and efficiency there.
- Options for the user (Phase 4 report): refit on more speed lines, add a high-Mach loss, or flag/limit high tip speed.

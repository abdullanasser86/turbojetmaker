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

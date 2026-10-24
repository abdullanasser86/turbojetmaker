# Project brief for Claude Code: ML-driven micro turbojet design & optimization

> Paste this whole file as your first message to Claude Code in an empty project folder (or save it as `PROJECT_BRIEF.md` in the repo and tell Claude Code: "Read PROJECT_BRIEF.md and start Phase 0").

---

## 1. Mission

Build **TurboML**, a specialized (non-LLM) machine-learning system that performs **preliminary design and multi-objective optimization of micro turbojet engines in the 50–1000 N thrust class**.

The system must:
1. Model the engine with physics (thermodynamic cycle + 1-D meanline component models).
2. Generate training data from those physics models.
3. Train fast ML surrogate models (with uncertainty) that predict performance and feasibility.
4. Use the surrogates inside optimizers to find Pareto-optimal designs.
5. **Always verify final designs with the physics models**, never report surrogate-only results.

Scope of this first version: **cycle level + component level (meanline)**. 3-D CFD/FEA and generative blade geometry are explicitly out of scope for now, but keep the architecture open for adding them later (as an extra "fidelity level").

Reference architecture (typical of this thrust class, e.g. KJ-66 type engines):
- Single spool
- Single-stage **centrifugal compressor** with vaned (or vaneless) radial diffuser
- Annular combustor
- Single-stage **axial turbine** (NGV + rotor)
- Fixed convergent nozzle

## 2. About me (the user) — how to work with me

- I am **new to both ML and propulsion**. Teach as you build:
  - Every physics module gets a module docstring explaining the physics in plain language (what the component does, which equations, which assumptions).
  - Maintain `docs/LEARNING.md`: a running glossary (TSFC, pressure ratio, TIT, slip factor, surge, Pareto front, GP, hypervolume, etc.) and a short "what we did and why" log per phase.
  - Each phase ends with a short Jupyter notebook in `notebooks/` that I can run to see results and plots.
- **Stop at the end of every phase** and give me: what was built, test/validation results, open issues, and the decisions you need from me. Do not start the next phase until I say go.
- When there is a genuine design choice (e.g. which loss model set, which objective to prioritize), give me 2–3 options with a recommendation and trade-offs, then wait.
- Be honest about uncertainty. If a model can't be validated against real data, say so clearly. **Never invent validation data, engine specs or paper results.** If you need data you don't have, tell me what to find (paper, dataset) and I'll supply it.

## 3. Compute & environment

- Single GPU workstation (CUDA). Use the GPU for neural-network training and BoTorch; physics sims run on CPU in parallel (`multiprocessing` / `joblib`).
- Python 3.11 (check compatibility of all packages below before pinning; drop to 3.10 only if something requires it).
- Use `uv` or `conda` with a **pinned** lock file. Verify `torch.cuda.is_available()` in Phase 0.

## 4. Pre-researched state of the art (verify versions at start — this research was done Oct 2026)

Before writing code in Phase 0, **check current versions / install status** of each tool below (PyPI / GitHub) and report any discrepancies.

### 4.1 Cycle analysis
- **pyCycle** (NASA, built on OpenMDAO): 1-D steady-state thermodynamic cycle analysis with analytic derivatives, CEA-based thermo, design and off-design modes. Ships with a `simple_turbojet.py` example — start from it. Citation: Hendricks & Gray, "pyCycle: A Tool for Efficient Optimization of Gas Turbine Engine Cycles", *Aerospace* 6(87), 2019. Repo: github.com/OpenMDAO/pyCycle.
- **OpenMDAO** is actively released (3.4x series as of 2026). pyCycle pins compatible OpenMDAO versions per release — check the compatibility table in the pyCycle README and pin accordingly.
- Also write a **small independent 0-D cycle model** (pure NumPy, variable cp or simple two-gamma model) for cross-checking pyCycle and for very fast low-fidelity data. Disagreement between the two beyond tolerance = bug to investigate.

### 4.2 Centrifugal compressor meanline
- **RadComp** (`pip install radcompressor`, github.com/cyrilpic/radcomp): open-source radial compressor meanline model adapted from Schiffmann & Favrat (Energy, 2010); used to generate the DATED synthetic engineering dataset (Picard, Schiffmann & Ahmed, 2023). Evaluate it as the starting point; extend or wrap it rather than writing from scratch if it fits.
- Loss model literature to implement / choose from:
  - **Aungier** (*Centrifugal Compressors: A Strategy for Aerodynamic Design and Analysis*, ASME Press 2000) — the standard comprehensive meanline loss set (incidence, blade loading, skin friction, clearance, mixing, disk friction, recirculation, leakage, choke).
  - **Oh et al.** tested published loss models against experiments and proposed an "optimum" combination (Galvas-type losses + Aungier choke loss).
  - **Slip factor**: Wiesner is standard but known to be inaccurate for modern high-speed impellers (Harrison, Purdue PhD 2020: Wiesner predicted only ~half of cases within ±0.05). Treat slip factor as an **uncertain parameter** (input to uncertainty quantification), not a fixed truth.
- Optional: `ccp` (centrifugal compressor performance library, Lüdtke-based, CoolProp gas properties) for cross-checks.

### 4.3 Axial turbine meanline
- **TurboFlow** (`pip install turboflow`, github.com/turbo-sim/turboflow, MIT license; Anderson et al., *JOSS* 10(111), 7588, 2025): meanline axial turbine model with on/off-design performance analysis and design optimization, equation-oriented, gradient-based solvers, selectable loss/deviation/choking submodels, CoolProp properties. Evaluate it first.
  - Check whether it handles combustion-product gas (it uses CoolProp fluids; air may be an acceptable approximation at cycle level, but document the error). If unsuitable, write a compact meanline turbine model with Kacker–Okapuu or Ainley–Mathieson/Dunham–Came losses.

### 4.4 Combustor, nozzle, mechanical constraints (simple models)
- Combustor: total pressure loss fraction + combustion efficiency + fuel-air ratio from energy balance (or via pyCycle's combustor). Kerosene/Jet-A as fuel.
- Nozzle: convergent, choked/unchoked check.
- Mechanical feasibility (simple, documented proxies):
  - Impeller tip speed limit from material (aluminum vs titanium) — research and justify limits, make them config parameters.
  - Turbine blade centrifugal stress via AN² proxy and disk burst margin proxy.
  - TIT / turbine material temperature limit (creep proxy).
- These are preliminary-design proxies. Label them as such in code and outputs.

### 4.5 Validation data sources (real-world reality checks)
- Review paper: "A review on the structures and characteristics of micro-turbojet engines" (Sharif University of Technology; J. of Solid and Fluid Mechanics) — includes specs of **100+ micro-turbojets below 1000 N** (thrust, dimensions, rotor speed, air flow, fuel consumption, turbine temperature). Use this as an **engine database** to (a) set realistic design-variable bounds, (b) sanity-check optimizer outputs (are we in a plausible region?), (c) compare thrust/TSFC trends. Ask me to obtain the PDF if you can't access it.
- KJ-66 compressor: Xiang, Schlüter & Duan (2017), *Proc. IMechE Part G* 231(5) — CFD + experiment; reported peak pressure ratio rising from ~1.54 to ~1.96 as speed goes 80→117 krpm, with peak efficiency dropping from ~0.73 to ~0.55. Use as a compressor-model validation case.
- AFRL/DTIC thesis on small commercial turbojet overall and component-level performance (apps.dtic.mil, AD1100321).
- SR-30 teaching turbojet papers (published thrust/TSFC/component efficiency measurements).
- De Villiers, Stellenbosch MEng thesis 2014: meanline design + testing of a micro gas turbine centrifugal compressor.
- If you need a dataset I haven't provided, list it and I will try to get it.

### 4.6 ML / optimization state of the art to use
- **Surrogates**
  - **Gaussian Processes** (GPyTorch / BoTorch `SingleTaskGP`, `ModelListGP`): default for < ~5k samples; give calibrated uncertainty.
  - **Deep ensembles of MLPs** (PyTorch, GPU): for larger datasets / many outputs; ensemble spread = uncertainty.
  - **Feasibility classifier** (gradient boosting or MLP): predicts "physics model converges and design is feasible" (no surge, no choke, constraints met). Failed/non-converged simulations are **training data for this classifier**, not discarded.
  - **Multi-fidelity GP** (Kennedy–O'Hagan autoregressive / discrepancy approach): low fidelity = 0-D cycle with assumed efficiencies; high fidelity = cycle coupled to meanline components. Use in Phase 5+ to save expensive evaluations.
- **Optimization**
  - **pymoo** (NSGA-II for 2–3 objectives, NSGA-III for 4+; constraints as `G <= 0`) on the surrogate.
  - **BoTorch multi-objective Bayesian optimization** with **qLogNEHVI** (numerically stable noisy expected hypervolume improvement; supports outcome constraints — see BoTorch tutorial `constrained_multi_objective_bo`) for the active-learning loop against the true physics model.
  - Track **hypervolume** per iteration as the main convergence metric.
- **Active learning**: a 2025 AIAA SciTech paper (JuliaSim, arXiv 2501.07701) showed active learning produced a turbofan surrogate with ~0.1% relative error, outperforming brute-force quasi-Monte Carlo sampling. Use active learning (sample where uncertainty or expected improvement is highest) rather than only brute-force sampling.
- **Dataset design**: follow the DATED guidelines (Picard et al. 2023) for synthetic engineering datasets — document sampling, bounds, failure rates, and coverage.
- **Sensitivity analysis**: Sobol indices (SALib) on the surrogate to show which design variables matter most.
- RL is *not* the primary method here; skip it.

## 5. Problem definition (initial — refine with me in Phase 1)

### Design variables (bounds to be derived from the engine database + literature, then confirmed with me)
- Cycle: compressor total pressure ratio, air mass flow, turbine inlet temperature (TIT), shaft speed.
- Compressor: impeller tip radius, inducer hub/tip radii, blade count (integer), backsweep angle, exit blade height, diffuser radius ratio / vane count, tip clearance.
- Turbine: mean radius, blade height, flow coefficient, loading coefficient, degree of reaction (or equivalent geometric parameters).
- Combustor: pressure loss fraction (parameter, possibly fixed).
- Nozzle: exit area.

### Objectives (multi-objective)
- Minimize **TSFC** (fuel efficiency).
- Maximize **thrust-to-weight** (needs a simple mass estimation model — implement a documented empirical/geometric mass model) or **thrust per frontal area**.
- Optional third: minimize max diameter or rotor tip speed (robustness / manufacturability).

### Constraints
- Thrust within target band (configurable, e.g. user picks 100 N, 200 N, 500 N...).
- Compressor surge margin ≥ configurable value; no choke at design point.
- Impeller tip speed ≤ material limit; turbine AN² ≤ limit; TIT ≤ material limit; EGT ≤ limit.
- Turbine power = compressor power / mechanical efficiency (shaft balance) — enforced by the solver.
- Envelope: max diameter and length (configurable).

### Operating points
- Design point: static, sea-level ISA (Mach 0).
- At least one off-design check: e.g. idle speed and one flight condition (e.g. Mach 0.5 at 3 km, configurable). A design that fails off-design is infeasible.

## 6. Repository architecture

```
turboml/
  pyproject.toml / uv.lock (or environment.yml)
  CLAUDE.md                 # conventions for future Claude Code sessions (you create this)
  PROJECT_BRIEF.md          # this file
  configs/                  # YAML configs (bounds, constraints, materials, operating points)
  turboml/
    units.py                # SI everywhere; helper conversions
    gas/                    # gas properties (wrappers; air + combustion products)
    physics/
      cycle_pycycle.py      # pyCycle turbojet wrapper
      cycle_simple.py       # independent 0-D NumPy cycle (cross-check, low fidelity)
      compressor.py         # centrifugal meanline (RadComp-based or own, Aungier/Oh losses)
      turbine.py            # axial meanline (TurboFlow-based or own)
      combustor.py, nozzle.py
      mechanical.py         # tip speed, AN², mass model, envelope
      engine.py             # coupled engine: matching, design + off-design, returns a result dataclass
    data/
      sampling.py           # Sobol / LHS, integer handling
      generate.py           # parallel dataset generation, failure capture, parquet output
      dataset_card.md       # auto-generated summary
    surrogates/
      gp.py, ensemble.py, feasibility.py, multifidelity.py
      evaluate.py           # metrics, calibration, OOD detection
    optimize/
      problem.py            # pymoo problem wrapping surrogates
      nsga.py               # surrogate-based NSGA-II/III
      mobo.py               # BoTorch qLogNEHVI active-learning loop with real physics
      verify.py             # re-run Pareto designs through physics
    analysis/
      sensitivity.py, plots.py, design_card.py
  validation/
    cases/                  # validation inputs + reference data (with source citations)
    run_validation.py
  tests/                    # pytest: unit + regression + physics sanity tests
  notebooks/                # one per phase, for the user
  docs/
    LEARNING.md
    DECISIONS.md            # architecture decision log
    VALIDATION.md           # what was validated, against what, with what error
```

## 7. Phased plan with acceptance criteria

### Phase 0 — Setup & verification (stop after)
- Verify tool versions (section 4); install; pin; GPU check.
- Scaffold repo, `CLAUDE.md`, pre-commit (ruff, black), pytest, config system.
- Run pyCycle's `simple_turbojet` example unmodified and report output.
- Install and run a hello-world for RadComp and TurboFlow.
- **Accept when**: environment reproducible from lock file, all example runs succeed, report written.

### Phase 1 — Cycle model (stop after)
- Adapt pyCycle turbojet to micro scale (single spool, centrifugal compressor represented by map/efficiency, realistic component efficiencies & losses for this size class — cite sources).
- Build the independent 0-D NumPy model.
- **Accept when**: the two models agree within ~1% on thrust and TSFC for matched assumptions across a parameter sweep; results for a few published engines from the database are within a plausible band (document errors honestly — component efficiencies are unknown, so this is a sanity check, not validation); unit tests cover energy/mass balance.

### Phase 2 — Component meanline models (stop after)
- Centrifugal compressor meanline with selectable loss sets; generate speed lines / compressor map; surge & choke detection.
- Axial turbine meanline; generate map.
- **Accept when**: compressor model reproduces KJ-66-type trends (pressure ratio and efficiency vs speed) within a documented error; turbine model reproduces at least one published case; slip-factor uncertainty is propagated and shown.

### Phase 3 — Coupled engine model (stop after)
- Couple components into the cycle: shaft speed + power balance + flow continuity matching; design mode (size components) and off-design mode (fixed geometry, find operating point).
- Robust solver: good initial guesses, failure detection with a clear failure reason (`surge`, `choke`, `no_convergence`, `constraint_X`).
- **Accept when**: a full design + off-design run for a baseline ~100–200 N engine completes in seconds; results are physically plausible vs the engine database; failure modes are labelled.

### Phase 4 — Dataset generation (stop after)
- Sobol sampling over the full design space; parallel generation; store inputs, all outputs, failure flags and reasons; dataset card with coverage and failure statistics.
- Start with ~2k–5k samples; scale up only if surrogate accuracy needs it.
- **Accept when**: dataset + card exist; failure rate reported; a held-out test set is frozen.

### Phase 5 — Surrogates (stop after)
- Train and compare: GP vs deep ensemble for each output (thrust, TSFC, mass, surge margin, etc.); feasibility classifier; optionally multi-fidelity GP.
- Report R², RMSE, max error, **uncertainty calibration** (coverage of predicted intervals), and performance on held-out data. Explain results in LEARNING.md.
- **Accept when**: key outputs (thrust, TSFC) reach agreed accuracy (target: < 2% median relative error, I'll confirm) with calibrated uncertainty; feasibility classifier precision/recall reported.

### Phase 6 — Optimization + active learning (stop after)
- NSGA-II on surrogates → candidate Pareto front.
- BoTorch qLogNEHVI loop calling the real physics model (batch size tuned to CPU cores), retraining surrogates each iteration; track hypervolume.
- Verify every final Pareto design with physics (design + off-design). Report surrogate vs physics discrepancy for each.
- **Accept when**: hypervolume has converged (plateau), verified Pareto front exists, designs are within plausible region of the engine database (or deviations are explained).

### Phase 7 — Analysis & reporting (stop after)
- Pareto plots (thrust/weight vs TSFC), parallel-coordinate plots, Sobol sensitivity, "design card" per selected engine (all key dimensions, stations' temperatures/pressures, margins, uncertainties).
- **Accept when**: notebook produces the full report end-to-end from config.

### Later (not now)
- Add 3-D CFD (SU2) for selected designs as a higher fidelity level in the multi-fidelity model; neural operators / GNNs for field prediction; parametric blade geometry and generative design.

## 8. Engineering rules

- SI units internally; units in every variable name or docstring where ambiguous.
- Every physics function: docstring with equations, assumptions, and source citation.
- Deterministic seeds; every run logged (config + git hash + results) — use a simple run directory structure or MLflow.
- Tests: physics sanity (energy/mass conservation, ideal limits: with 100% efficiencies the model must match the ideal Brayton cycle analytically), regression tests for validated cases, surrogate smoke tests.
- Configs, not hard-coded numbers. Material limits, bounds, operating points all in YAML.
- Never silently clip, drop or "fix" failed simulations; record them.
- Never report surrogate predictions as final results. Final = physics-verified.
- Keep `docs/DECISIONS.md` updated with every significant choice and its reason.

## 9. Scope note

This is a **preliminary/conceptual design** tool. Its outputs are not manufacturing-ready: real micro turbines spin at 100k+ rpm and need detailed 3-D aero, structural, rotordynamic, thermal and combustion analysis plus physical testing by qualified people. State this in the README and on every design card.

## 10. Start now

Begin with **Phase 0**. First, give me a short plan for Phase 0 and a list of anything you need from me. Then execute it and stop at the end of Phase 0 with your report.

---

### References (gathered beforehand)
- pyCycle: https://github.com/OpenMDAO/pyCycle — Hendricks & Gray, Aerospace 6(87), 2019, doi:10.3390/aerospace6080087
- OpenMDAO release list: https://openmdao.org/newdocs_list/
- RadComp: https://pypi.org/project/radcompressor — https://github.com/cyrilpic/radcomp
- TurboFlow: https://github.com/turbo-sim/turboflow — JOSS doi:10.21105/joss.07588
- ccp: https://ccp-centrifugal-compressor-performance.readthedocs.io
- Harrison, slip factor PhD (Purdue 2020): https://docs.lib.purdue.edu/dissertations/AAI30504084
- Turbocharger compressor loss-mechanism meanline (Oh / Aungier discussion): https://openaccess.city.ac.uk/id/eprint/22490/8/energies-12-02711.pdf
- Micro-turbojet review (100+ engines < 1000 N): https://jsfm.shahroodut.ac.ir/article_3053.html?lang=en
- KJ-66 compressor study: https://journals.sagepub.com/doi/10.1177/0954410016644632
- AFRL small turbojet performance thesis: https://apps.dtic.mil/sti/pdfs/AD1100321.pdf
- Micro gas turbine centrifugal compressor design (Stellenbosch): https://scholar.sun.ac.za/items/523e5647-52c7-4c38-a2c6-5c81887795ce/full
- Active-learning jet engine surrogate (JuliaSim): https://arxiv.org/html/2501.07701
- BoTorch constrained MOBO tutorial: https://botorch.org/tutorials/constrained_multi_objective_bo
- Multi-fidelity GP for gas turbine combustor optimization (Toal et al.): https://eprints.soton.ac.uk/444749/
- pymoo: https://pymoo.org

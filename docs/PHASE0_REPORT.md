# Phase 0 report — setup & verification (2026-10-24)

## Acceptance criteria
| Criterion | Status |
|---|---|
| Environment reproducible from lock file | ✅ `uv.lock` (152 packages); a fresh clone + `uv sync --frozen` rebuilt it and the full test suite passed |
| pyCycle `simple_turbojet` runs unmodified | ✅ 3 points (DESIGN, OD0, OD1) converge in ≈13 s, but only with the pyCycle git pin + numpy<2.4 (D-003, D-004) |
| RadComp hello-world | ✅ notebook case + air speed sweep |
| TurboFlow hello-world | ✅ one-stage example converges; upstream regression suite matches in all physical outputs |
| GPU check | ⚠️ `torch.cuda.is_available() = False`: this cloud container has **no GPU**. Torch is the CUDA 13.0 build, so it must be re-checked on your workstation (`uv run python scripts/check_env.py`) |
| Repo scaffold, CLAUDE.md, pre-commit, pytest, config system | ✅ 11 tests pass; ruff + black clean |

## Tool versions vs. the brief
| Tool | Brief said | Found (PyPI, 2026-10-24) | Notes |
|---|---|---|---|
| pyCycle (`om-pycycle`) | – | 4.4.0 (Oct 2025) | **Broken with numpy 2 / OpenMDAO ≥3.42** (issue #116). Fix merged on master May 2026, unreleased → pinned commit `ee7e161`. README compat table only lists up to 4.2.0. |
| OpenMDAO | 3.4x series | 3.45.1 (Sep 2026) | ✓ matches brief |
| RadComp (`radcompressor`) | – | 1.0.1 (May 2025) | 1 of its unit tests fails on an exact float comparison under CoolProp 8 (647.0959999999873 vs 647.096): harmless |
| TurboFlow | – | 0.1.18 (Jun 2025) | Declares `numpy<2`, `CoolProp<7` → overridden (D-002). **Also has a centrifugal-compressor model** and an HECC compressor validation case, which the brief doesn't mention |
| ccp (`ccp-performance`) | optional | 0.4.1 (Sep 2026) | not installed (optional; it depends on `ctREFPROP`). Can be added in Phase 2 if wanted |
| torch | – | 2.14.1 | the PyPI Linux wheel bundles CUDA 13.0 |
| BoTorch / GPyTorch | – | 0.18.1 / 1.15.2 | ✓ |
| pymoo | – | 0.6.2 | ✓ |
| SALib | – | 1.6.0 | needs numpy ≥2 (part of why the TurboFlow override was needed) |
| CoolProp | – | 8.0.0 | used by RadComp + TurboFlow |

## Example results (plumbing checks, not validation)
- **pyCycle** (full-size engine): DESIGN Fn = 11 800 lbf (52.5 kN), OPR 13.5, TSFC 0.79852 lbm/(lbf·h)
  (22.6 g/(kN·s)). OD0 and OD1 converge as well. Full output: `validation/phase0/pycycle_simple_turbojet.txt`.
- **RadComp**: Olmedo & Schiffmann stage 2 (ammonia, the notebook case): PR 1.4976, η 0.6953. Air sweep of the
  Schiffmann & Favrat impeller from 150 to 260 krpm: PR 1.00→1.51, η rising with speed.
- **TurboFlow**: Kofskey (1972) single-stage turbine, air: η_ts 79.12 %, η_tt 92.57 %, ṁ 2.737 kg/s.

## Findings to carry into later phases
1. **Speed budget.** Costs per evaluation:

   | Model | Time per evaluation |
   |---|---|
   | RadComp | ≈0.2 s per point |
   | TurboFlow | ≈3–7 s per point |
   | pyCycle example | ≈13 s for 3 points, including model setup |

   A 5 000-sample dataset at 3 operating points through TurboFlow alone would take roughly 4–7 h on
   4 cores. That's acceptable, but it's a factor in the Phase 2 choice between TurboFlow and a
   compact in-house turbine meanline.
2. **Fluids.** RadComp and TurboFlow get gas properties from CoolProp. Air works; combustion products (turbine) are
   not a standard CoolProp fluid. The error of treating turbine gas as air has to be quantified in Phase 2, as the brief
   anticipates.
3. **RadComp's validity flag isn't sufficient.** At 150 krpm it returns "valid" with PR = 1.000, η = 0.006.
   We need our own physical-sanity checks, which will produce labelled failure reasons.
4. **TurboFlow's centrifugal-compressor module** is a second, independent compressor meanline model. It's useful as a cross-check
   on RadComp in Phase 2.

## What's in the repo
`pyproject.toml` + `uv.lock` · `CLAUDE.md` · `configs/default.yaml` (operating points from the brief; all
bounds and limits `null` until researched) · `turboml/{config,units,runlog}.py` · `scripts/check_env.py` ·
`scripts/phase0/*` · `tests/` · `notebooks/00_environment_check.ipynb` (executed) · `docs/{LEARNING,DECISIONS,VALIDATION}.md` ·
`third_party/` (verbatim upstream example inputs, with licences).

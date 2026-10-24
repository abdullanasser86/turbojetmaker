# Validation log

**Nothing in TurboML has been validated against measured engine data yet.**

Terminology used in this project:
- **Regression check** – "the code gives the same answer as last time / as the upstream
  reference". Detects breakage, says nothing about physical truth.
- **Verification** – "the code solves its equations correctly" (e.g. ideal Brayton limit,
  energy/mass balance, agreement between two independent models).
- **Validation** – "the model agrees with reality" (measured data, with stated error).
- **Sanity check** – "the result is in a plausible range" (e.g. versus an engine database
  when component efficiencies are unknown). Weaker than validation.

## Phase 0 — regression baselines (environment checks only)

| Tool / case | Quantity | Our value (2026-10-24) | Reference | Kind |
|---|---|---|---|---|
| pyCycle `simple_turbojet` DESIGN (full-size ~52 kN engine) | Fn / TSFC | 11800.0 lbf / 0.79852 lbm/(lbf·h) | Fn is the example's design target | regression |
| pyCycle `simple_turbojet` OD0 / OD1 | Fn / TSFC | 11000.0 / 0.78364 · 8000.0 / 0.82951 | Fn are example targets | regression |
| TurboFlow upstream regression suite (7 configs) | all physical outputs | agree to ≥6 significant figures | TurboFlow stored Linux references | regression |
| TurboFlow one-stage example (Kofskey 1972 geometry) | η_ts / ṁ | 79.12 % / 2.7371 kg/s | none yet (experimental data available in TurboFlow `validation_cases/` → Phase 2) | regression |
| RadComp notebook case (Olmedo & Schiffmann stage 2, NH₃) | PR / η | 1.4976 / 0.6953 | none (notebook's stored output used REFPROP and a different case → not comparable) | regression |

Regression values are asserted in `tests/test_external_tools.py`.

## Phase 1 — cycle model

### Verification (code solves its equations correctly)
| Check | Result | Where |
|---|---|---|
| Ideal Brayton limit (perfect gas, no losses, full expansion): η_th = 1 − PR^(−(γ−1)/γ) | matches to 1e-9 (PR 1.5, 2.0, 2.4) | `tests/test_cycle_simple.py` |
| Whole-engine energy balance incl. fuel, mech. loss, off-take | closes to 1e-10 | `tests/test_cycle_simple.py` |
| Shaft power balance, mass balance (W9 = W + Wf) | exact | `tests/test_cycle_simple.py` |
| Species cp, S°, ΔHf vs JANAF reference values (N2, O2, Ar, CO2, H2O) | within 0.01 J/mol/K, 20 J/mol | `tests/test_thermo.py` |
| LHV round trip of fuel enthalpy definition | exact | `tests/test_thermo.py` |
| ISA vs pyCycle's US-1976 table at 0/3/6 km | < 1e-4 (after geometric→geopotential fix) | `tests/test_atmosphere.py` |
| **pyCycle (CEA) vs 0-D, 64-point Sobol sweep** (ground–6 km, M 0–0.6, wide efficiency ranges) | Fn median 0.03 %, TSFC median 0.06 %; **63/64 within 1 %**; worst point Fn 1.03 % / TSFC 0.93 % (ill-conditioned: specific thrust 36 N·s/kg at M 0.5; 0.18 % of gross thrust) | `scripts/phase1/cross_check_sweep.py`, `validation/phase1/cross_check*.{csv,txt}` |
| pyCycle (TABULAR) vs 0-D, same sweep | Fn median 0.34 % / max 3.2 %; TSFC max 3.5 %; compressor work max 1.0 % (table interpolation) | same |

**Acceptance (user decision 2026-10-24): ACCEPTED at 63/64 within 1 %.** The metric was not changed afterwards.
Re-run with the parallel cross-check module (CEA only, 70 s): identical numbers; 1/64 flagged (1.6 %).

**Secondary metric (labelled; context only, never pass/fail):** |ΔFn|/Fg, median 0.025 %, max 0.184 %.

#### Known exception: sample 8 (`validation/phase1/known_exceptions.yaml`, test `tests/test_crosscheck_known_exception.py`)
| | value |
|---|---|
| inputs | W 0.127 kg/s, PR 4.63, η_c 0.555, TIT 1108 K, η_t 0.726, η_m 0.951, M 0.50 at 2594 m |
| primary |ΔFn/Fn| / |ΔTSFC/TSFC| | 1.03 % / 0.93 % |
| secondary |ΔFn|/Fg | 0.18 % |
| nozzle pressure ratio | 1.09 (unchoked, barely above ambient) |
| Fg/Fn (ram-drag amplification) | 5.6 (sweep median 1.24; highest in sweep) |
| d ln Fn / d ln Pt9 | 31 (vs ~5–6 at typical points) |

Cause: two multiplying amplifiers (jet velocity at near-unity nozzle pressure ratio, and net = gross − ram drag) turn
the ~0.03 % turbine-pressure-ratio difference between CEA equilibrium chemistry and the 0-D complete-combustion model
into ~1 % in net thrust. |ΔFn/Fn| correlates with Fg/Fn at r = 0.92 across the sweep. Not a modelling error. The
regression test fails if this point's difference leaves the band 0.8–1.3 % or the secondary metric exceeds 0.3 %.
Such points (nearly thrustless, poor components) lie far outside realistic designs. The 5 % gate still counts them.

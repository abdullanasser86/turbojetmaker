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

## Planned validation cases (to be done in their phases)
- Phase 1: cycle model vs. independent 0-D model (verification); published engines from the
  micro-turbojet review database (sanity check — **PDF needed from user**).
- Phase 2: KJ-66 compressor, Xiang, Schlüter & Duan (2017) — **paper/data needed from user**;
  Kofskey (1972) single-stage turbine (data shipped with TurboFlow).

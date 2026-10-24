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

### Sanity check vs published engines (NOT validation)
Data: 5 JetCat engines, 63–425 N (`validation/cases/jetcat_datasheets.yaml`). **Unverified web-search
extracts** of manufacturer pages; listed fuel flow and SFC are mutually consistent within 0–3 % for an assumed fuel density
of 0.80 kg/L.

- *Forward*, with provisional baseline assumptions (η_c 0.72, η_t 0.80, TIT 1150 K, η_b 0.95):
  thrust error −19 % … +23 %, **SFC under-predicted by 10–15 % for every engine**. The placeholder efficiencies are optimistic.
- *Inverse* (solve TIT and η_t to match thrust + SFC exactly for each assumed η_c):
  - P160-RXi-B: consistent for any η_b ≤ 1 (implied EGT 691 °C at η_b 0.95; listed max 750 °C).
  - P62-RX: consistent if η_b ≤ 0.93 (implied EGT 742 °C at η_b 0.95 vs listed max 730 °C).
  - For both: implied TIT 1090–1130 K and η_t 0.62–0.77 for η_c 0.82…0.64.
  - P220-RXi: implied EGT ≈ 787 °C at η_b 0.95 (no EGT listed).
  - **P300-PRO, P400-PRO: inconsistent** at η_b 0.95. Implied EGT 933 / 978 °C vs listed max 750 °C, which would
    need η_b ≤ 0.74 / 0.70. Since EGT follows from energy conservation (FAR = thrust·SFC/W), this points at the
    extracted numbers (mass flow? thrust?) or at the EGT definition (sensor reading vs mean total temperature),
    not at component efficiencies. **Needs the official datasheets.**

## Data requests (user)
1. Micro-turbojet review PDF ([R] in configs/bounds_proposal_phase1.yaml): full engine table.
2. Official JetCat datasheet PDFs for P62-RX, P160-RXi-B, P220-RXi, P300-PRO, P400-PRO, to verify the extracted
   values and find the EGT definition.
3. Any source with measured or CFD-derived component efficiencies for 100–500 N engines (compressor η, turbine η,
   combustor ΔP/P and η_b). Candidates surfaced by search: the AFRL thesis AD1100321; the JetCat P300-RX integrated
   model study (Lee et al., 2024); the KJ-66 compressor paper (Xiang et al., 2017).

## Planned validation cases (to be done in their phases)
- Phase 1: cycle model vs. independent 0-D model (verification); published engines from the
  micro-turbojet review database (sanity check — **PDF needed from user**).
- Phase 2: KJ-66 compressor, Xiang, Schlüter & Duan (2017) — **paper/data needed from user**;
  Kofskey (1972) single-stage turbine (data shipped with TurboFlow).

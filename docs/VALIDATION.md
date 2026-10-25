# Validation log

**No whole-engine validation yet.** Components have been checked against measured component data (KJ-66 compressor,
Kofskey turbine); the coupled engine has only been verified (consistency) and sanity-checked against databases.

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

## Phase 2 — component meanline models

### Compressor (TurboML meanline, `turboml/physics/compressor.py`)
**Verification**
| Check | Result | Where |
|---|---|---|
| Loss-free, σ = 1, radial blades, frictionless diffuser: Δh0 = U2², η = 1 | exact (1e-9) | `tests/test_compressor.py` |
| Port of the Oh (1997) loss set vs TurboFlow's impeller (same conventions), 9 points 70–110 krpm | PR within 0.013 %, η within 0.004 pts | `tests/test_compressor.py`, scratch study |
| TurboFlow slip hook (degrees, no NaN at −40°) | pass | `tests/test_compressor.py` |

**Validation vs KJ-66 MEASURED data** (Murray 2009, digitized from Xiang et al. 2016; ±0.003 kg/s, ±0.01 PR, ±0.005 η)
| Model | PR error mean / max abs | η error mean / max abs |
|---|---|---|
| TurboML, blind (published geometry + labelled assumptions) | +20.9 % / 33.8 % | +13.3 / 15.6 pts |
| RadComp, blind (independent Schiffmann loss set; 14/32 points failed) | +11.6 % / 18.0 % | +3.2 / 5.8 pts |
| TurboML, calibrated on 90 krpm only → **held-out 70/80/100/110 krpm** | **+1.1 % / 4.6 %** | **+0.9 / 3.1 pts** |

- Calibrated effective parameters: Δσ = −0.104 (σ = 0.70), diffuser Cf = 0.094. They absorb the unknown backsweep and
  the unmodelled wedge-vane diffuser and 90° turn.
- **Trends reproduced:** PR rises steeply with speed; peak η stays almost flat (0.754 → 0.770 over 70–117 krpm), as
  measured (0.73–0.77). The CFD η drop (0.73 → 0.55, Xiang) is neither measured nor reproduced.
- **Slip uncertainty:** Δσ = ±0.05 shifts PR by ≈ ±5 %. Bands are shown in notebook 02.
- **Not validated:** surge (no surge data); choke location (no choke data); diffuser model form (real diffuser is vaned).

### Axial turbine (TurboFlow, `turboml/physics/turbine.py`)
**Validation vs Kofskey & Nusbaum (1972), NASA TN D-6967, MEASURED, no tuning** (50–110 % speed, PR 1.6–4.6)
| Quantity | mean | mean abs | max abs | n |
|---|---|---|---|---|
| η_ts | −1.18 pts | 1.42 pts | 2.83 pts | 106 |
| mass flow | +1.13 % | 1.13 % | 2.69 % | 46 |
| torque | −0.52 % | 1.48 % | 5.19 % | 60 |
| exit flow angle | −0.1° | 1.78° | 6.63° (110 % speed) | 50 |

**Gas model error** (air instead of combustion products, TIT 1000–1250 K, FAR 0.015–0.025, PR 1.8–2.6): turbine work
−0.17 … −0.46 %, choked flow function +0.28 … +0.47 %. TurboML air vs CoolProp air: cp within 0.15 %.

## Phase 3 — coupled engine model

### KJ-66 compressor calibration: calibration / validation split (user decision 2026-10-25)
| Set | Speeds | Rows | Use |
|---|---|---|---|
| Calibration | 90 krpm | 5 PR + 5 efficiency | fit Δσ and diffuser Cf; 2σ band from the LSQ Jacobian |
| Validation (held out) | 70, 80, 100, 110 krpm | all other measured rows | the only rows on which errors are reported |
Band (statistical, no model-form error): Δσ −0.1041 ± 0.0074 (1σ), Cf 0.0937 ± 0.0039 (1σ), correlation −0.14.
Source and file: `configs/components/kj66_compressor_calibration.yaml`, `scripts/phase3/compressor_calibration_band.py`.

### Fast axial turbine (`turboml/physics/turbine_meanline.py`)
**Verification (port) vs TurboFlow 0.1.18, Kofskey geometry, 278 common map points (50–110 % speed, PR 1.6–4.6)**
| mode | mass flow max abs | η_ts max abs | power max abs |
|---|---|---|---|
| `turboflow` sign (default) | 0.099 % | 0.061 pts | 0.123 % |
| `rotor_plus` (TurboFlow's intended rotor sign) | 1.7 % | 4.3 pts | 7.5 % |
Benner loss port vs TurboFlow's own function: equal to 1e-9 (`tests/test_turbine_meanline.py`).

**Verification across the design space vs TurboFlow** (`scripts/phase3/turbine_port_verification.py`; 256 scrambled
Sobol samples, seed 20261024, each sized from φ 0.35–0.9, ψ 0.9–2.2, R 0.2–0.6, AR 0.8–2.0, clearance 0.5–5 %,
work 80–250 kJ/kg, mass flow 0.08–1.9 kg/s, speed from the database trend ×0.75–1.3, T0 1000–1350 K, p0 1.9–4.8 bar;
evaluated at the estimated design point and at one off-design point (speed 60–110 %, PR ×0.7–1.5)). Air in both
models. Acceptance: η_ts ±2 pts, mass flow ±2 %, work ±2 %.
| | mean abs | 95th pct abs | max abs | outside acceptance |
|---|---|---|---|---|
| η_ts | 0.050 pts | 0.109 pts | 0.471 pts | 0 |
| mass flow | 0.048 % | 0.108 % | 0.151 % | 0 |
| specific work | 0.043 % | 0.109 % | 0.651 % | 0 |
454 points compared (229 samples). 27 samples gave no geometry (hub radius ≤ 0: flow coefficient too low for the flow
and speed), recorded as `sizing_failed`. 4 points failed in TurboFlow only. **Accepted.**
Worst cases: all at off-design part speed (63–82 %) with a choked stator and positive rotor incidence (8–23°). They are
mostly at high φ (0.6–0.9) and low reaction (0.2–0.4), i.e. where the inner loss iteration has to work hardest.
TurboFlow takes 18.8 s per point (median). The fast model takes 34 ms per cold point while 3 TurboFlow processes run on
the same 4 cores (~17 ms uncontended, 7.5 ms in sweeps).

**Validation vs Kofskey MEASURED data (untuned)**
| mode | η_ts mean / max abs | mass flow mean / max abs | torque mean / max abs |
|---|---|---|---|
| `turboflow` (default) | −1.20 / 2.84 pts | +1.08 / 2.59 % | −0.53 / 5.06 % |
| `rotor_plus` | −2.00 / 3.72 pts | +0.76 / 2.03 % | −2.38 / 9.40 % |
Per speed, η_ts mean error (`turboflow` / `rotor_plus`): 50 %: +0.13 / −2.63; 70 %: −1.33 / −2.66; 90 %: −1.89 / −1.98;
100 %: −1.73 / −1.73; 110 %: −1.11 / −1.10 pts. The rotor sign only matters at large rotor incidence (low speed).

**Not validated at micro scale:** the Kofskey turbine is ~0.24 m in diameter, Re ~ 4e5. No measured micro-turbine
(r ~ 25–35 mm) data is available yet, and the baseline's η_tt = 0.895 is probably optimistic.

### Coupled engine (`turboml/physics/engine.py`), baseline ~150 N design
**Verification**
| Check | Result |
|---|---|
| Off-design solver at the design speed reproduces the design point | Fn, TIT within 1e-6 (`tests/test_engine.py`) |
| Design point vs 0-D `cycle_simple` with the coupled model's component efficiencies | Fn 0.000 %, TSFC 0.000 % |
| Design point vs pyCycle-CEA, same inputs | Fn −0.02 %, TSFC −0.075 % (flag tolerance 1 %) |
| Power balance, mass balance, nozzle flow at every converged point | residuals < 1e-8 |

**Sanity check vs engine databases** (review table engines 104–193 N, n = 32; JetCat PRO 2024). Not validation: the
databases have no component data for this geometry.
| Quantity | Model | Database |
|---|---|---|
| specific thrust | 494 N s/kg | 352–511 (engines < 250 N) |
| TSFC | 41.5 g/(kN s) | 38.0 / 46.4 / 54.5 (min/median/max) |
| shaft speed | 130 krpm | 115 / 126.5 / 175 krpm; trend 131 krpm |
| diameter (proxy) | 115 mm | 90 / 103 / 120 mm |
| length (proxy) | 298 mm | 198 / 283 / 346 mm |
| EGT | 759 °C | 680 / 700 / 760 °C; JetCat max 720–750 °C |
| idle (35 % speed) | 6.4 % thrust, EGT 654 °C | JetCat idle 29–33 % speed, 3.3–5.7 % thrust, EGT min 480 °C |
The idle EGT is about 170 K hotter than JetCat's minimum EGT. Part-speed component efficiencies and the loss-scaling
assumption are the main suspects (Phase 3 report, open issues).

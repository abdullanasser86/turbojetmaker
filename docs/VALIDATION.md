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

## Phase 4 — creep data, calibration band, speedup, dataset

### NiDI No. 393 rupture data (`validation/cases/ni393_cast_rupture.csv`, `turboml/physics/creep.py`)
- Transcription: values read from 300-dpi renders of the scanned tables (the PDF has no text layer). The ksi and
  printed MPa columns agree within 2.5 % for every entry, except one documented misprint (713C, 1000 h, 1400 °F:
  458 MPa printed vs 65 ksi = 448 MPa); the ksi column is used throughout (`tests/test_mechanical.py`).
- Values marked "extrapolated" in the booklet are flagged in the CSV.
- Interpolation reproduces every tabulated point exactly and is monotonic in temperature and life. The
  Larson–Miller fit (used only for lives other than 100/1000 h) has a maximum error of 10 % (713C) and 7.7 %
  (MAR-M 247) on the tabulated points.
- Not validated: the check uses *typical* rupture data (not design minima) and a cantilever stress proxy. This is a
  ranking tool, labelled as a proxy.

### Compressor calibration band (`scripts/phase4/compressor_band_inflation.py`)
| inflation k | held-out points inside the envelope (of 34) |
|---|---|
| 1 (statistical 2σ) | 14 (41 %) |
| 2 | 28 (82 %) |
| 2.5 | 32 (94 %) |
| **2.75 (chosen)** | **34 (100 %)** |
Caveat: the held-out points now set the band width, so they are no longer independent evidence for it.

### Compressor speedup (verification, `scripts/phase4/profile_engine.py` → `validation/phase4/profiling.yaml`)
- The Newton diffuser solve and the original bracketed solve give the same results: max relative output difference
  3e-11 on the baseline engine (design + 3 off-design points) and 7e-14 over 40 compressor points, with identical
  failure labels.
- Speed: 3.3–4.3× faster per engine evaluation.

### Dataset `phase4_v1` pilot (256 designs)
- Reproducibility: regenerating the pilot at commit e2f2a36 gave bit-identical statuses, feasibility, convergence
  flags and thrusts compared with the first run.
- Cross-check (CLAUDE.md gate), coupled model vs pyCycle-CEA on a random 2 % of designed engines (2 of 92): 0 flagged;
  max |ΔFn| 0.039 %, max |ΔTSFC| 0.067 %; gate passed. The 0-D `cycle_simple` vs pyCycle-CEA also had 0 flagged.
- Two designs is a thin sample. The full run checks 2 % of ~700 designed engines (~14).
- Baseline regression: the Phase 3 baseline still gives 148.286 N after the Phase 4 changes (`tests/test_engine.py`).

## Phase 4 (round 2) — KJ-66 rebuild check, idle EGT, dataset v2 pilot

### KJ-66 compressor (`scripts/phase4/kj66_rebuild_check.py` → `validation/phase4/kj66_rebuild_check.yaml`)
Geometry: the user's Xiang 2017 extract matches the model exactly; no calibration factor can be removed (D-047).
Recalibration on the 90 krpm measured points reproduces Δσ = −0.1041, Cf = 0.0937.

| speed | model peak η | measured max η (MEAS) | Xiang CFD peak η | model PR at peak η | measured max PR | Xiang CFD peak PR |
|---|---|---|---|---|---|---|
| 70 krpm | 0.754 | 0.770 | – | 1.365 | 1.392 | – |
| 80 krpm | 0.757 | 0.770 | 0.73 | 1.495 | 1.520 | 1.54 |
| 90 krpm | 0.761 | 0.768 | – | 1.658 | 1.695 | – |
| 100 krpm | 0.765 | 0.767 | 0.63 | 1.855 | 1.865 | 1.76 |
| 110 krpm | 0.768 | 0.752 | – | 2.097 | 2.075 | – |
| 117 krpm | 0.770 | – | 0.55 | 2.293 | – | 1.96 |

- Manufacturer points (MFR; error floor 10 %):
  - Xiang design PR 2.2 at 0.22 kg/s, 117 krpm → model 2.341 (+6.4 %).
  - Murray PR ≈ 2.54 at 128 krpm → model 2.67 at the peak-efficiency flow (+5 %; extrapolated beyond the 110 krpm
    data).
- Above 100 krpm the model's speed lines have no interior PR peak above 0.05 kg/s, so the surge side is not resolved
  there.

### Idle EGT (`scripts/phase4/idle_egt_investigation.py` → `validation/phase4/idle_egt_investigation.yaml`)
EGT is compared as a bare-wire thermocouple reading, T_s + 0.68 (T_t − T_s) (Murray, ASSUM), at the turbine exit.
KJ-66 replicas use the KJ-66 compressor family; the turbine stage choices (φ 0.7, ψ 1.0, R 0.5) are an ASSUMPTION
because the KJ-66 turbine geometry is not available.

| engine | matched to (MFR) | TIT | EGT reading, full power | spec EGT | EGT at 35 % / 28 % speed |
|---|---|---|---|---|---|
| KJ-66 Xiang build | 75 N at 0.22 kg/s, PR 2.2, 117 krpm | 996 K | 908 K (635 °C) | 843 K | 887 / 893 K |
| KJ-66 Murray build | 92.3 N and SFC 0.154 kg/(N h), PR 2.54, 128 krpm (W solved 0.260 kg/s) | 957 K | 849 K (576 °C) | 863 K (590 °C) | 847 / 854 K |
| Phase 3 baseline | – | 1150 K | 1024 K | – | 926 / 934 K |

Step by step, for the baseline at 35 % speed:
1. **Efficiencies:** idle η_c 0.70 (design 0.75), η_t,ts 0.72 (0.74). Sensitivity: −10.4 K idle EGT per +1 pt of
   η_c, −10.1 K per +1 pt of η_t.
2. **Thermocouple:** the reading is 6–9 K below the total temperature at full power and ≤ 1 K below it at idle.
3. **Combustion efficiency at idle** 0.95 → 0.90 → 0.80: EGT −1.1 / −3.6 K, fuel flow +5.6 / +18.8 %. In a
   power-balanced, speed-controlled engine the turbine sets the TIT, so η_b changes the fuel flow, not the
   temperatures.
4. **Reynolds number** at idle is 0.26 (compressor) and 0.27 (turbine) of design. A Reynolds correction LOWERS idle
   efficiencies, so it would RAISE idle EGT; it is the wrong direction. No correlation was applied (none cited).

What this shows:
- The original comparison (baseline idle 654 °C vs KJ-66 full-power 590 °C) mixed two engines.
- On KJ-66 replicas, full-power EGT is within −14 to +65 K of the manufacturer values.
- But the model's EGT hardly falls between full power and idle (ratio 0.98–1.01 on the replicas). JetCat's idle-to-max
  EGT bands are 190–330 K wide (P20 210, P80-SE 190, P220 330, P300–P550 270 K). These are operating bands, not idle
  values (MFR), but they indicate a much cooler low end.
- None of steps 1–4 closes a gap of that size: the efficiencies would need about +10 pts each at idle.
- The remaining suspect is the low-speed matching: compressor flow/PR at 28–35 % speed, which lies below the 70 krpm
  calibration data, and turbine flow capacity at PR ≈ 1.1. Testing it needs Murray's measured 48–64 krpm PR, mass flow
  and TIT (Figs 5.1–5.3, figures only).

### Dataset `phase4_v2` pilot (`scripts/phase4/pilot_feasibility_analysis.py`)
- 200 designs: 15 designed, **11 feasible (5.5 % < 10 %): stopped** (D-049).
- Cross-check: 1 of 15 designed engines checked, 0 flagged (ΔFn 0.010 %, ΔTSFC 0.054 %).
- The pilot ran from the working tree that was committed as af26454. The later diagnostic hook does not change
  results (test `test_diag_hook_off_is_identical_and_on_moves_egt_the_right_way`; Phase 3 baseline outputs unchanged).

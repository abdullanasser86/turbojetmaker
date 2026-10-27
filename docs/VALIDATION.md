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

## Phase 4 (round 3) — Murray (2009) as the second KJ-66 validation case; option-A pilot

Data: `validation/cases/murray2009/`.
- Appendix D tables are TRANSCRIBED: MEAS, or calculated from MEAS by Murray. This is the exhaust-diffuser
  configuration, 48–120 krpm.
- Appendix C run plots are DIGITIZED [MEAS], including the three turbojet (nozzle) runs down to 28 krpm. Digitizer
  error against Appendix D (same run): T5 rms 5.8 °C, P03 rms 0.41 kPa.
- Script: `scripts/phase4/murray_validation.py` → `validation/phase4/murray2009_*`. Figs 5.1–5.3 / 5.12–5.19 were not
  in the upload.

### Compressor (KJ-66 model, stored 90 krpm calibration; Appendix D at the measured speed, flow and inlet state)
| set | n | PR error mean / rms / max | efficiency error mean / rms / max | within Murray's uncertainty (PR / η) |
|---|---|---|---|---|
| validation | 13 | +0.1 / 2.1 / 5.2 % | +0.6 / 3.0 / 7.4 pts | 5/13 / 12/13 |
| near calibration (88, 91 krpm) | 2 | 0.0 / 0.8 / 0.8 % | 0.0 / 1.6 / 1.7 pts | 2/2 / 2/2 |
| validation below 70 krpm (first data there) | 5 | −1.4 / 1.4 / 1.6 % | +0.9 / 3.9 / 7.4 pts | |
- PR is −1.6 … −1.1 % at 48–64 krpm and +3 … +5 % at 112–120 krpm (`murray2009_compressor.png`).
- The model's efficiency is almost flat (0.745–0.762). The measured efficiency drops at 120 krpm (0.706) and at the
  second 48 krpm point (0.671, ±0.086); the model misses both.
- Murray's PR uncertainty (±0.019) is tighter than the model error at most points, hence the 5/13.

### Turbine (INDICATIVE: assumed replica stages, KJ-66 turbine geometry unknown)
Sized at the 120 krpm point (φ 0.7, R 0.5, ψ 1.0/1.5/2.0), evaluated at all 15 points:
- flow capacity +5.4 … +5.7 % (mean);
- efficiency **+10.0 … +13.1 pts** above Murray's derived η_tt (0.667–0.823, ±0.07–0.37).

This is the first measured hint on the "micro-scale turbine efficiency" flag: a meanline model with Kofskey-scale
correlations looks ~10 pts optimistic for a KJ-66-size turbine. It is not conclusive: the geometry is assumed, and
Murray's turbine efficiency is derived from the compressor work.

### Turbojet engine and idle EGT (nozzle runs 1–3; TIT fitted at 120 krpm only; thermocouple reading T_s + 0.68 (T_t − T_s))
| ψ (assumed) | EGT error, rising / falling speed | 104–120 krpm | 62–96 krpm | 48–56 krpm | 28–32 krpm | P03 error | mass flow error |
|---|---|---|---|---|---|---|---|
| 1.0 | +49 / +70 K | +25 K | +55 K | +79 K | +71 K | −5.2 kPa | −8.2 % |
| 1.5 | +32 / +49 K | +18 K | +36 K | +53 K | +41 K | −5.1 kPa | −8.5 % |
| 2.0 | +34 / +50 K | +18 K | +38 K | +56 K | +45 K | −5.1 kPa | −8.7 % |

What the measurements show (`murray2009_egt_vs_speed.png`):
- The real KJ-66 turbojet's EGT reading is nearly flat, 480–535 °C over 48–120 krpm, and rises again to ~535 °C at
  28–32 krpm.
- On run 3 the falling-speed leg reads 20–40 K cooler than the rising leg (thermal soak). Run 2 does not show this
  consistently, and runs differ by up to ~45 K at the same speed.

What the model does:
- It has the same U shape, but too hot: +18 K near the fitted point, growing to +53–79 K at 48–56 krpm.
- Its part-speed operating line has 8–9 % less air and ~5 kPa less delivery pressure.
- Its compressor characteristic is within ~2 % at the measured flow, so the difference is in the matching (turbine
  and nozzle flow capacity at low pressure ratio).
- It fails at 27.6 krpm (impeller-exit choke at very low speed and flow, a model limit; recorded).

Conclusion on the idle-EGT question:
- The real engine IS hot at idle. Its EGT at 28–48 krpm (485–538 °C) is about the same as at 120 krpm.
- Our model over-predicts low-speed EGT by ~40–80 K, not by the 150–250 K the round-2 comparison suggested.

### Dataset `phase4_v3` pilot (option A)
- 200 designs: 51 designed, **17 feasible (8.5 % < 10 %): stopped** (D-054).
- Cross-check: 2 checked, 0 flagged (max ΔFn 0.03 %, ΔTSFC 0.12 %).

### Windows (D-050)
- `scripts/phase4/windows_smoke_test.py` PASSES in the Linux container: a spawn pool and loky are identical to
  serial, and the generator with 2 workers writes and resumes.
- `tests/test_windows_env.py` checks the lock and the win32 CUDA check. Not yet run on Windows itself.

## Phase 4 (round 4) — Murray Chapter 5: measured KJ-66 maps and operating lines

Data:
- `validation/cases/murray2009/murray2009_ch5_maps_digitized.csv` (Figs 5.12/5.14/5.16/5.18).
- `murray2009_ch5_operating_lines_digitized.csv` (Figs 5.1–5.3).
- All points are digitized [MEAS] with u_dig and u_pub (D-057).
- Script `scripts/phase4/murray_maps_validation.py` → `validation/phase4/murray2009_maps_*`, plot `murray2009_maps.png`.

### Compressor (stored calibration, no refit; standard-day inlet so model flow/speed are corrected values)
| corrected speed | set | PR error (mean) | corrected-flow offset (mean) | efficiency error (mean / rms) | within published uncertainty (PR / η) |
|---|---|---|---|---|---|
| 40 krpm (1 pt) | validation | −1.7 % | −4.7 % | +14.5 pts | 1/1 / 0/1 |
| 50 | validation | −1.1 % | −3.1 % | +0.4 / 3.6 pts | 2/2 / 5/5 |
| 60 | validation | −1.2 % | −1.5 % | −0.3 / 3.3 pts | 1/1 / 5/5 |
| 70 | validation | −1.1 % | −0.6 % | −1.5 / 3.2 pts | 1/1 / 3/4 |
| 80 | validation | −0.7 % | −0.3 % | −2.0 / 3.1 pts | 2/2 / 3/5 |
| 90 | near calibration | +0.1 % | 0.0 % | −1.0 / 2.6 pts | 2/2 / 3/4 |
| 100 | validation | +2.0 % | +0.6 % | +0.8 / 1.3 pts | 2/2 / 3/3 |
| **110** | validation | **+4.1 %** | +4.5 % | +2.3 / 2.5 pts | **0/2** / 1/3 |
| **120** | validation | **+7.1 %** | +6.1 % | **+6.1** / 6.1 pts | **0/2** / **0/2** |
| all validation lines | | +1.4 % (rms 3.4) | +0.7 % (rms 3.7) | +0.7 / 4.3 pts | 9/13 / 20/28 |

- The corrected-flow offset is the closest approach of the model speed line to the measured point, with distances
  scaled by the published uncertainties. Inverting PR(m) is ill-conditioned where the model's speed lines are flat.
- **Contradiction with the calibration at 110–120 krpm** (D-058): a systematic over-prediction of PR and efficiency
  beyond the published uncertainty. Reported; NOT refitted.
- The single 40 krpm point (η 0.60 ± 0.09) is far below the model's 0.74. Murray notes low-speed efficiencies vary by
  ~10 % from run to run.

### Turbine (INDICATIVE: assumed replica stages, φ 0.7, R 0.5; KJ-66 geometry unknown)
| replica ψ | corrected flow error mean / rms | per speed line (20 → 68 krpm) | efficiency error mean / rms |
|---|---|---|---|
| 1.0 | +6.3 / 8.4 % | +19 % → +1 % | +12.3 / 12.8 pts |
| 1.5 | +5.4 / 7.2 % | +18 % → +1 % | +11.6 / 12.2 pts |
| 2.0 | +5.2 / 7.1 % | +18 % → +1 % | +9.9 / 10.5 pts |
- The shape of the flow-capacity curve and the choking at high PR are reproduced; the error grows at low PR.
- The efficiency is ~10–12 pts above Murray's derived values at every speed line (published uncertainty ±0.08–0.34).
- This agrees with round 3: the meanline turbine looks optimistic at KJ-66 size, but it is not conclusive with an
  assumed geometry.

### Engine operating lines and idle check (turbojet replicas from round 3, TIT fitted at 120 krpm only)
| ψ | PR vs all-run envelope (Fig 5.1) | corrected flow vs envelope (Fig 5.2) | TIT vs the same nozzle run (Fig 5.3), all / below 60 krpm |
|---|---|---|---|
| 1.0 | −0.016 mean (rms 0.018) | −4.7 % (rms 5.4) | +40 K (rms 47) / +64 K |
| 1.5 | −0.015 (rms 0.017) | −5.0 % (rms 5.7) | +23 K (rms 29) / +39 K |
| 2.0 | −0.015 (rms 0.017) | −5.2 % (rms 5.9) | +25 K (rms 31) / +41 K |

Idle check against Fig 5.3:
- Murray's turbojet TIT rises from 780–820 K at 48 krpm to 920–940 K at 119 krpm (+100–140 K).
- The model's TIT is flat at 840–860 K below ~70 krpm and rises only ~60–70 K.
- So the model's low-speed TIT is 25–65 K too high. This is the same size as the low-speed EGT over-prediction of
  round 3 (+40–80 K), so the excess heat is in the matched TIT, not in the EGT reading.
- The engine runs slightly to the left of the measured operating line (−5 % corrected flow, −0.015 PR). The
  compressor speed lines are within ~1–2 % in that region, so this points to the hot-section matching at low PR.
- The indicative turbine check also shows the model's low-PR turbine characteristic departing from the measured one
  (+18 % flow capacity at PR < 1.12). How that maps onto the engine depends on the real turbine geometry, which is
  unknown, so the turbine/nozzle low-PR behaviour remains the prime suspect rather than a proven cause.

## Phase 4 (round 5) — compressor refit, KJ-66 turbine calibration, idle EGT, Nickel 337

Calibration and validation sets are kept separate everywhere; errors are model − measurement.

### Compressor refit R1 (`scripts/phase4/compressor_refit_murray.py` → `validation/phase4/compressor_refit_murray.*`)
Murray Figs 5.12/5.14 (digitized [MEAS]), standard-day inlet. Calibration lines 50/70/90/110 krpm; held out
40/60/80/100/120 krpm. Δσ −0.1176, Cf 0.1002 (D-059).

| line [krpm] | U2 [m/s] | set | PR err [%] | flow err [%] | η err [pts] | PR / η within published unc. |
|---|---|---|---|---|---|---|
| 40 | 138 | held out | −2.0 | −5.9 | +14.2 | 1/1, 0/1 |
| 50 | 173 | cal | −1.5 | −4.5 | +0.1 | 2/2, 5/5 |
| 60 | 207 | held out | −1.8 | −2.2 | −0.6 | 1/1, 4/5 |
| 70 | 242 | cal | −1.9 | −1.1 | −1.7 | 1/1, 3/4 |
| 80 | 277 | held out | −1.6 | −0.7 | −2.3 | 2/2, 3/5 |
| 90 | 311 | cal | −1.2 | −0.5 | −1.2 | 2/2, 3/4 |
| 100 | 346 | held out | +0.6 | +0.2 | +0.6 | 2/2, 3/3 |
| 110 | 380 | cal | +2.4 | +2.5 | +2.2 | 0/2, 1/3 |
| 120 | 415 | held out | **+5.0** | **+4.4** | **+6.1** | 0/2, 0/2 |

Held-out lines together: PR rms 2.8 % (stored calibration 3.8 %), flow rms 3.2 % (3.7 %), η rms 5.0 pts (5.0). The
trend with speed remains (model-form). Growth law and extrapolation band: D-060.

### KJ-66 turbine calibration (`scripts/phase4/turbine_calibration_murray.py` → `validation/phase4/turbine_calibration_murray.*`)
Geometry from published counts (Xu et al. 2021, KJ66 variant) + calibrated, not measured hub/tip and angles (D-061/062).

| fit | data fitted | hub/tip | NGV g [°] | rotor g [°] | flow err per line [%] | V5 err [%] | η err [pts] |
|---|---|---|---|---|---|---|---|
| **F (adopted)** | flow + V5 | 0.743 (0.65–0.745) | 64.8 | 43.6 | +11 (20) … −10 (68) | −2 (cal), −3 (held out); 11/15 within unc. | **+11.2** (all), +10.9 (≤ 44 krpm) |
| Q | flow | 0.52 ± 0.60 | 70.9 | 67.9 | −2.8 … +4.3 | −47 | +12.9 |
| FE | flow + η | 0.40 (bound) | 76.2 | 51.6 | −8.8 … +9.4 | −38 | +2.5 |

**Low-speed turbine efficiency does NOT match Murray** with any geometry consistent with the flow and exit-velocity
data: the model is 11–13 pts above Murray's (derived, ±0.08–0.34) efficiency on every line from 32 to 68 krpm. Only
the implausible FE geometry (hub/tip at its bound, 48° exit swirl, V5 −38 %) brings it down.

### Idle EGT re-check: KJ-66 replica (`scripts/phase4/kj66_replica_idle_egt.py` → `validation/phase4/kj66_replica_idle_egt.*`)
Real compressor geometry (R1), KJ-66 turbine (F or Q), nozzle area A9 fitted per run at the top speed (EGT reading,
r = 0.68); every other speed is a prediction (38 points, nozzle runs 1–3).

| turbine | EGT err all [K] | EGT err < 50 krpm [K] | model TIT < 50 krpm [K] | EGT within unc. | flow err [%] | P03 err [kPa] |
|---|---|---|---|---|---|---|
| F | +52 (rms 59) | +78 | 877 | 3/38 | −14 | −1.8 |
| Q | **−13 (rms 18)** | **−13** | **785** | 15/38 | −11 | −2.1 |
| round 4 (replica stages) | | +40 … +80 | 840–860 | | | |

Murray's calculated TIT at 48 krpm is 780–820 K (Fig 5.3): turbine Q reproduces it; turbine F is ~60–90 K hot.
Diagnostic (not a model change): lowering the turbine efficiency by the calibration gap makes the low-speed EGT hotter
(F +128 K, Q +53 K), so the hot idle of the earlier replicas was not caused by the optimistic turbine efficiency; it
follows the turbine flow capacity (F's capacity is 4–10 % low above 50 krpm corrected, Q's within ±4 %).
Caveats: both replicas under-predict air flow by 3 % (top) to 13–17 % (low speed) at nearly the right delivery
pressure. The model's speed lines are almost flat (120 krpm: PR 2.40 → 2.29 over 0.12–0.26 kg/s), so the engine's air
flow is set by the turbine + nozzle swallowing capacity; the idle-EGT agreement of Q may partly be compensating errors.
Q's geometry also contradicts the measured exit velocity (V5 −47 %).

### Nickel 337 (`scripts/phase4/ni337_rupture_crosscheck.py` → `validation/phase4/ni337_crosscheck.yaml`)
- vs Nickel 393: identical at all 7 common (T, life) points. User anchors at 1350 °F: 76 ksi (1000 h), 56 ksi
  (10,000 h), (97) ksi at 100 h extrapolated — all as printed.
- AMS 5391 (≥ 30 h at 1800 °F / 22 ksi, as cast): Table II test 65 h; p. 7 table 72 h; LM curve 67 h; creep model 49 h.
  **Pass.**
- Larson–Miller (Fig 7, digitized) vs the individual tests: rms 10.8 %; the 2000 °F tests lie 12–37 % below the master
  curve, so extending the tables with the LM curve would be non-conservative. Not done.
- Creep model vs Nickel 337 at 100 h: exact (same tables). At 10 h, 2000 °F the model's LM fallback is +32 %
  (non-conservative; not used at the 100 h design life).
- **Missing data or physics** (TIT > 1150 K, infeasible designed engines): v4 141 physics / 16 missing data; v5 131 / 16.
  Physics = AN² at the rotor-relative temperature fails for both alloys (rotor metal 1086–1314 K, inside the data);
  31–33 of them would also melt an IN-713C vane (hot streak above the 1533 K solidus). Missing data = MAR-M 247 NGV
  above its 1311 K data (NGV hot streak 1342–1588 K); Nickel 337 (IN-713C only, ≤ 2000 °F) cannot resolve them.

### Dataset `phase4_v5` (full, 2000 designs)
Card `docs/dataset_cards/phase4_v5.md`. 604 feasible (30.2 %); validated tip speed 40.2 % of designs feasible,
extrapolated 22.6 %; TIT > 1150 K 1.0 %. Cross-check vs pyCycle-CEA: 21 checked, 0 flagged (max |ΔFn| 0.037 %,
|ΔTSFC| 0.094 %).

## Phase 4 (round 6) — R2 test (Galvas), turbine loss multiplier, KJ-66 turbine variants

### R2: Galvas (1973) NASA TN D-7487 (`scripts/phase4/galvas_r2_test.py` → `validation/phase4/galvas_r2_test.yaml`)
Galvas has no shock or Mach-dependent loss (D-065). Its inducer choke criterion (choking incidence vs rms relative
Mach, listing p. 26/29), applied to the KJ-66 (rms blade angle 45.9°, assumed from the digitized 42° midspan):

| line [krpm] | M_W,rms | rms incidence [°] | choking incidence [°] | margin [°] | choked points |
|---|---|---|---|---|---|
| 90 | 0.52–0.53 | +14.9 … +16.5 | −15.6 | 29.9 | 0 |
| 100 | 0.58–0.59 | +14.2 … +15.8 | −10.8 | 24.2 | 0 |
| 110 | 0.65–0.66 | +13.1 … +15.0 | −6.8 | 19.2 | 0 |
| 120 | 0.71–0.73 | +12.5 … +14.5 | −3.3 | 14.9 | 0 |

Below 90 krpm M_W,rms < 0.5 (outside the table). Held-out compressor errors with the criterion = without (D-059
table unchanged). phase4_v5: 4 of 1031 designed engines would be flagged inducer-choked (all extrapolated).

### Turbine loss multiplier k_loss (`scripts/phase4/turbine_loss_multiplier_fit.py` → `validation/phase4/turbine_loss_multiplier.*`)
| variant | k_loss | interval | held-out η err [pts] k=1 → fit | held-out flow err [%] k=1 → fit | failed points at fit |
|---|---|---|---|---|---|
| T-A (hub/tip 0.743) | 1.955 | 1σ 1.8–2.0 | +11.3 → **+1.7** (rms 3.5) | −5.7 → −9.4 | 0 |
| T-B (hub/tip 0.519) | 1.762 | 2σ 1.6–1.7 | +12.8 → **+6.6** (rms 7.2) | −1.7 → −5.9 | 5 (68 krpm line) |

Per held-out line (T-A, fitted): 32: +1.4 · 44: +1.8 · 56: +3.8 · 68: −1.7 pts; all within Murray's published
uncertainty. The 26 krpm calibration point stays −12 pts (single point, Murray η 0.85 ± 0.34).

### KJ-66 turbine variants in the replica (`scripts/phase4/kj66_turbine_variants.py` → `validation/phase4/kj66_turbine_variants.*`)
Replica as in round 5 (real compressor + R1, A9 fitted per run at the top speed). Run 2 (120 krpm top):

| variant, k_loss | idle EGT err (< 50 krpm) | idle TIT | Fn @ 120 krpm | TSFC @ 120 krpm [g/kN/s] | η_t,ts @ 120 | air flow err @ 120 |
|---|---|---|---|---|---|---|
| T-A, 1 | +78 K | 877 K | 56.1 N | 43.5 | 0.658 | −8 % |
| T-B, 1 | −13 K | 785 K | 53.9 N | 47.1 | 0.787 | −3 % |
| T-A, 1.955 | +109 K (run 2 only) | 897 K | 47.4 N | 50.4 | 0.608 | −10 % |
| T-B, 1.762 | −0.3 K (run 1 only, top 112 krpm) | 815 K | – | – | – | – |

With the fitted k_loss the replica's top-speed point does not converge (`turbine_no_convergence`) for 2 of 3 runs
per variant (T-A runs 1 and 3, T-B runs 2 and 3) at any nozzle area; at k = 1 all runs converge. Spread T-A vs T-B at
k = 1: thrust 4 %, TSFC 8 %, idle EGT ~90 K; the turbine efficiency differs most (0.66 vs 0.79 total-to-static:
T-A's short, choked NGV and fast exit flow). Murray's thrust is not in our data, so thrust is not validated.
Hobbyist hint (hub/tip ≈ 0.56, hobbyist CAD, unverified): closer to T-B.

## Phase 4 (round 7) — turbine-solver restarts, phase4_v6, surrogates on v6

### Turbine-solver restarts (D-067; `validation/phase4/retry_recovery_check.yaml`)
| v5 pilot (200 designs) re-evaluated | recovered on retry | still failing | converged points changed |
|---|---|---|---|
| k_loss 1 (v5 settings) | 0 (no turbine-solver failures existed) | 0 turbine; 97 design-mode, 1 idle compressor choke | none (max \|ΔFn\| = 0) |
| k_loss 1.86 (v6 nominal) | 1 | 8 `turbine_no_convergence` (6 cruise, 2 part speed) | – |

In the full v6: 3 points converged on a restart (2 design, 1 cruise); 44 failed after all restarts (37 cruise and
6 part-speed `turbine_no_convergence`).

### phase4_v6 (`docs/dataset_cards/phase4_v6.md`)
- 2000 designs: 1028 designed, 568 feasible (28.4 %; v5 30.2 %).
- Feasible by tip speed: validated 38.0 %, extrapolated 21.0 % of designs. TIT > 1150 K: 1.3 %.
- Turbine η_t,tt at design: 0.62–0.86 (median 0.789); 0 outside [0.58, 0.93] (Murray-based range, D-069).
- pyCycle-CEA cross-check: 21 checked, 0 flagged (max |ΔFn| 0.042 %, |ΔTSFC| 0.112 %).
- Feasible design TSFC: best 43.2, median 72.5 g/kN/s. JetCat claims 40–52 [MFR].
- Off-design convergence (designed engines):

| point | converged | main failure reasons |
|---|---|---|
| design | 1028 / 1028 | – |
| M 0.5 / 3 km | 946 / 1028 | 38 no net thrust, 37 turbine non-convergence |
| 80 % speed | 1015 / 1028 | 6 turbine non-convergence |
| idle (35 %) | 1018 / 1028 | 6 turbine limit loading |

### Surrogates on v6 (`validation/phase5/surrogates_phase4_v6.yaml`, comparison `compare_phase4_v5_vs_phase4_v6.md`)
Frozen v6 test set, deep ensemble:

| target | median relative error |
|---|---|
| Fn design | 0.60 % |
| Wf design | 0.36 % |
| Wf cruise | 0.58 % |
| Fn idle | 0.59 % |
| T05 idle | 0.30 % |

95 % coverage is 0.93–1.00. Feasibility classifier: accuracy 0.923, AUC 0.970, ECE 0.032. The TSFC targets have
heavy tails (low-thrust designs), so TSFC should be derived as Wf/Fn.

### Joint KJ-66 turbine refit (geometry + k_loss) and idle EGT (D-068)
`validation/phase4/turbine_joint_refit.yaml`, `kj66_joint_variants_idle_egt.yaml`.

| joint solution | hub/tip | k_loss | held-out η / flow err | V5 err (not fitted) | idle EGT err (< 50 krpm) | idle TIT | replica runs built |
|---|---|---|---|---|---|---|---|
| near T-A | 0.735 | 1.95 | +2.2 pts / −1.2 % | **−5 %** | +100 K | 917 K | 1 of 3 |
| middle (lowest RSS) | 0.649 | 2.33 | +1.7 pts / −1.1 % | −29 % | +87 K | 885 K | 3 of 3 |
| near T-B | 0.482 | 2.10 | +3.8 pts / −3.0 % | −51 % | **+29 K** | 845 K | 1 of 3 |

- The geometry that matches the measured exit velocity runs ~90–100 K hot at idle. The geometry closest to the idle
  EGT misses the exit velocity by half. **The joint refit does not resolve the conflict.**
- Since the turbine model cannot satisfy both with any geometry, the cause probably lies outside the turbine
  geometry. Candidates:
  - low-power combustion efficiency (fixed 0.95; Murray correlated it with fuel-air ratio);
  - the station-5 location assumption behind V5;
  - the replica's low air flow (−12 % at idle).
- Missing replica runs: top-speed `turbine_no_convergence` at the higher k_loss (same as round 6).
- Middle solution at 120 krpm (run 2): thrust 47.9 N, TSFC 53.5 g/kN/s, η_t,ts 0.68.

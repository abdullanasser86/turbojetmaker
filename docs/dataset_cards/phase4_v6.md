# Dataset card: `phase4_v6`

> **PRELIMINARY DESIGN ONLY - NOT MANUFACTURING-READY. Real micro turbines spin at 100 000+ rpm and need detailed 3-D aerodynamic, structural, rotordynamic, thermal and combustion analysis plus physical testing by qualified people before anything is built or run.**

Status: 2000 of 2000 planned designs evaluated.

Generated with git 6e7f9a6 by `scripts/phase4/generate_dataset.py` (`turboml/data/generator.py`), config `configs/dataset_phase4_v6.yaml`. Raw records: `datasets/<name>/designs.jsonl` (git-ignored, regenerate from seed + config).

## What one record is

One **design** = cycle design variables + turbine stage choices, sized by the coupled physics model (design mode), checked mechanically (creep, AN², tip speed), then evaluated at 4 **points**: design (SLS), M0.5 at 3 km (design speed), 80 % speed (SLS), idle 35 % speed (SLS). Every design and every point carries `converged` and `failure_reason`; failed designs and points are kept, never dropped.

Physics: 0-D cycle (`cycle_simple` components) with meanline compressor (KJ-66-calibrated, provisional) and meanline axial turbine (TurboFlow formulation); NOT surrogate output. Turbine efficiency at micro scale is unvalidated (flag on every design).

## Design variables (Sobol, scrambled, seed 20261117)

| variable | range | scale | source |
|---|---|---|---|
| W_air_kg_s | 0.08 – 1.9 | log | [U] from [JD] P500 0.90 kg/s @ 500 N, P550 0.93 @ 550 N; 50 N needs ~0.09 and 1000 N ~1.8 at 550-600 N s/kg. Database: e |
| impeller_tip_speed_m_s | 350 – 500 | linear | upper: user decision 2026-10-25 (<= 500 m/s, inside the 400-550 m/s structural range for turbocharger-type impellers, De |
| comp_specific_speed | 0.35 – 0.9 | linear | Phase 4 option A (user decision 2026-10-25), n_s,U = omega sqrt(Q01) / U2^1.5 (turboml.physics.engine.tip_speed_specific |
| tit_K | 1000 – 1350 | piecewise | [U2] User decision 2026-10-25 (pre-Phase 3): raised from 1250 K so AMT-class engines ([R]: EGT up to 875 C = 1148 K, TIT |
| turb_hub_tip_frac | 0 – 1 | linear | Phase 4 option T1 (user decision 2026-10-25): position inside the per-design buildable hub/tip range [0.5, min(0.95, hub |
| turb_psi | 0.9 – 2.2 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |
| turb_reaction | 0.2 – 0.6 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |

Compressor calibration factors: 397 of 2000 designs sample (Δσ, Cf) uniformly in the inflated band (`configs/components/kj66_compressor_calibration.yaml`); the others use the nominal values.

Fixed assumptions: component losses `configs/engine_baseline.yaml` (DePaola values as summarised by the user, off-design loss scaling with corrected flow²); materials and creep settings `configs/materials.yaml` (life 100 h, safety factor 1.5, blade taper At/Ah 1.0 (K = 1.00), pattern factor 0.25; NGV PF_max stored per design).

## Held-out test set (frozen)

`data/splits/phase4_v6_split.csv`: 15 % of design ids, seeded permutation, written before any design was evaluated. Never used for training or tuning.

| split (evaluated designs) | count |
|---|---|
| train_val | 1700 |
| test | 300 |

## Design-level outcome

| status | count |
|---|---|
| ok | 1028 |
| design_failed | 972 |

Design-mode failure reasons:

| reason | count |
|---|---|
| design_turbine_no_buildable_hub_tip | 957 |
| no_nozzle_pressure_ratio | 9 |
| design_turbine | 6 |

Mechanically feasible (selected material passes creep; tip speed OK): **568** of 1028 designed engines; premium material required: 77.

Infeasible designed engines by active constraint(s):

| active constraints | count |
|---|---|
| AN2 | 213 |
| AN2;TIT_ngv_above_material_data | 179 |
| TIT_ngv_above_material_data | 42 |
| creep_rotor | 18 |
| creep_rotor;TIT_ngv_above_material_data | 7 |
| TIT_rotor_above_material_data;TIT_ngv_above_material_data | 1 |

High-TIT band (~15 % of the samples by design, D-056): feasibility inside and outside it:

| band | designs | designed | feasible | % of designs | % of designed |
|---|---|---|---|---|---|
| TIT > 1150 K | 302 | 146 | 4 | 1.3 | 2.7 |
| TIT <= 1150 K | 1698 | 882 | 564 | 33.2 | 63.9 |

Infeasible high-TIT designs by active constraint (material status):

| active constraints (material status) | count |
|---|---|
| AN2;TIT_ngv_above_material_data (fails_creep) | 121 |
| TIT_ngv_above_material_data (unchecked_no_material_data) | 15 |
| creep_rotor;TIT_ngv_above_material_data (fails_creep) | 5 |
| TIT_rotor_above_material_data;TIT_ngv_above_material_data (unchecked_no_material_data) | 1 |

Infeasible high-TIT designs: PHYSICS or MISSING DATA (`turboml/analysis/feasibility.py`; physics = every material fails a check its rupture data cover, or IN-713C metal above its solidus; missing data = some material fails only because a metal temperature is above its data):

| cause | count |
|---|---|
| physics | 126 |
| missing_data | 16 |

Physics, IN-713C checks failed:

| checks | count |
|---|---|
| AN2 | 94 |
| AN2;ngv_above_solidus | 32 |

Missing data, data-limited material(s):

| materials | count |
|---|---|
| marm247 | 14 |
| in713c;marm247 | 2 |

Compressor tip-speed regime (R3, D-060): **validated** = U2 <= 415 m/s (top measured Murray speed line), **extrapolated** = above it (wider calibration band, one-sided toward lower slip / higher loss):

| regime / TIT band | designs | designed | feasible | % of designs | % of designed |
|---|---|---|---|---|---|
| validated / all TIT | 868 | 468 | 330 | 38.0 | 70.5 |
| validated / TIT <= 1150 K | 736 | 402 | 328 | 44.6 | 81.6 |
| validated / TIT > 1150 K | 132 | 66 | 2 | 1.5 | 3.0 |
| extrapolated / all TIT | 1132 | 560 | 238 | 21.0 | 42.5 |
| extrapolated / TIT <= 1150 K | 962 | 480 | 236 | 24.5 | 49.2 |
| extrapolated / TIT > 1150 K | 170 | 80 | 2 | 1.2 | 2.5 |

Design-point turbine total-to-total efficiency (sanity check, D-069): quantiles p0 0.6226, p5 0.7026, p25 0.7562, p50 0.7885, p75 0.813, p95 0.84, p100 0.8569; plausible range [0.58, 0.93] (Murray 2009 measured span +- published uncertainty); **0 designs flagged** (0 of them feasible), list in `validation/phase4/turbine_eta_flags_phase4_v6.csv`.

Turbine-solver restarts (D-067): points that converged only on a restart / failed after all restarts / recorded turbine_no_convergence:

| point | converged on retry | failed after retries | turbine_no_convergence |
|---|---|---|---|
| cruise_M05_3km | 1 | 38 | 37 |
| design | 2 | 0 | 0 |
| idle_35 | 0 | 0 | 0 |
| part_speed_80 | 0 | 6 | 6 |

Turbine loss multiplier (D-064): nominal vs sampled subset:

| subset | designs | designed | feasible | k_loss |
|---|---|---|---|---|
| nominal | 1603 | 838 | 454 | 1.86 – 1.86 |
| sampled | 397 | 190 | 114 | 1.6 – 2.0 |

Turbine hub/tip (option T1, D-055): sampled 0.5 – 0.852; the exit-Mach-0.5 limit was the binding upper bound for 1028/1028 designed engines; 957 designs had no buildable hub/tip (exit Mach > 0.5 even at 0.5).

Material status (designed engines):

| status | count |
|---|---|
| ok_default | 491 |
| fails_creep | 417 |
| ok_premium | 77 |
| unchecked_no_material_data | 43 |

Sensitivity (feasibility recomputed with one setting changed; the defaults above decide feasibility):

| setting | feasible designs |
|---|---|
| at_ah_0.34 | 831 |
| pf_0.15 | 604 |
| pf_0.35 | 439 |

Turbine outside DePaola 2020 Table 2 typical ranges (sanity flags only, not infeasible):

| quantity | designs flagged |
|---|---|
| an2_m2_rpm2 | 216 |
| loading_coefficient | 640 |
| exit_mach | 714 |
| exit_swirl_deg | 7 |

Derived by the sizing (not sampled): n_rpm 20553.4 – 272276; turb_phi 0.201 – 1.724; pr_c 1.665 – 3.245; comp_specific_speed_conventional 0.653 – 1.47

## Point-level outcome (designed engines)

| point | converged | failure reasons | surge flags |
|---|---|---|---|
| cruise_M05_3km | 946/1028 | no_net_thrust: 38, turbine_no_convergence: 37, no_convergence: line search failed: 3, no_nozzle_pressure_ratio: 2, no_convergence: max iterations: 1, turbine_limit_loading: 1 | – |
| design | 1028/1028 | – | – |
| idle_35 | 1018/1028 | turbine_limit_loading: 6, no_nozzle_pressure_ratio: 2, no_convergence: max iterations: 2 | – |
| part_speed_80 | 1015/1028 | turbine_no_convergence: 6, no_convergence: max iterations: 3, turbine_limit_loading: 2, no_nozzle_pressure_ratio: 2 | – |

## Cross-check (pyCycle-CEA, random 2 % of designed engines)

- checked 21 of 1028; coupled model vs pyCycle-CEA: flagged 0 (gate passed: True), max |ΔFn| 0.042 %, max |ΔTSFC| 0.1116 %
- 0-D cycle_simple vs pyCycle-CEA: flagged 0 (gate passed: True)
- secondary metric |ΔFn|/Fg (labelled secondary): max 0.00042

## Cost

Per design (one worker): mean 4.84 s, median 4.75 s, 95th percentile 11.12 s, max 413.1 s.
Last run: 1800 designs in 3048 s wall on 3 workers.

## Known limitations

- Compressor family is KJ-66-like (one geometry family, one calibration); vaneless diffuser model.
- Turbine efficiency at micro scale unvalidated; incidence-sign convention open (D-029).
- Creep check uses typical (not minimum) rupture data; untapered blades (At/Ah = 1, conservative); MAR-M 247 data stop at 1311 K.
- Turbine geometry outside TurboFlow's recommended ranges (kept, warning in `geom_turb_sizing_warnings`): rotor hub/tip in 0 of 1028 designed engines (feasible among them: 0); the loss correlations are less trustworthy there.
- A compressor operating on its choke line can't be resolved (labelled `choke_compressor_*`).
- Design variables are sampled independently in the box: many corners are physically impossible for this engine family (labelled design failures).

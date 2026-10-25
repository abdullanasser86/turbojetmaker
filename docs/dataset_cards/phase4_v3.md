# Dataset card: `phase4_v3` (PILOT)

> **PRELIMINARY DESIGN ONLY - NOT MANUFACTURING-READY. Real micro turbines spin at 100 000+ rpm and need detailed 3-D aerodynamic, structural, rotordynamic, thermal and combustion analysis plus physical testing by qualified people before anything is built or run.**

Status: 200 of 2000 planned designs evaluated (pilot: the first designs of the Sobol sequence).

**STOPPED by the pilot stop rule:** 17 of 200 pilot designs feasible (8.5 % < 10 %). The full run was not started. Analysis: `validation/phase4/pilot_analysis_phase4_v3.yaml`.

Generated with git f74c0b7 by `scripts/phase4/generate_dataset.py` (`turboml/data/generator.py`), config `configs/dataset_phase4_v3.yaml`. Raw records: `datasets/<name>/designs.jsonl` (git-ignored, regenerate from seed + config).

## What one record is

One **design** = cycle design variables + turbine stage choices, sized by the coupled physics model (design mode), checked mechanically (creep, AN², tip speed), then evaluated at 4 **points**: design (SLS), M0.5 at 3 km (design speed), 80 % speed (SLS), idle 35 % speed (SLS). Every design and every point carries `converged` and `failure_reason`; failed designs and points are kept, never dropped.

Physics: 0-D cycle (`cycle_simple` components) with meanline compressor (KJ-66-calibrated, provisional) and meanline axial turbine (TurboFlow formulation); NOT surrogate output. Turbine efficiency at micro scale is unvalidated (flag on every design).

## Design variables (Sobol, scrambled, seed 20261027)

| variable | range | scale | source |
|---|---|---|---|
| W_air_kg_s | 0.08 – 1.9 | log | [U] from [JD] P500 0.90 kg/s @ 500 N, P550 0.93 @ 550 N; 50 N needs ~0.09 and 1000 N ~1.8 at 550-600 N s/kg. Database: e |
| impeller_tip_speed_m_s | 350 – 500 | linear | upper: user decision 2026-10-25 (<= 500 m/s, inside the 400-550 m/s structural range for turbocharger-type impellers, De |
| comp_specific_speed | 0.35 – 0.9 | linear | Phase 4 option A (user decision 2026-10-25), n_s,U = omega sqrt(Q01) / U2^1.5 (turboml.physics.engine.tip_speed_specific |
| tit_K | 1000 – 1350 | linear | [U2] User decision 2026-10-25 (pre-Phase 3): raised from 1250 K so AMT-class engines ([R]: EGT up to 875 C = 1148 K, TIT |
| turb_hub_tip | 0.5 – 0.95 | linear | user decision 2026-10-25: rotor-exit hub/tip ratio inside TurboFlow's recommended range [0.5, 0.95] (turboflow geometry_ |
| turb_psi | 0.9 – 2.2 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |
| turb_reaction | 0.2 – 0.6 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |

Compressor calibration factors: 39 of 200 designs sample (Δσ, Cf) uniformly in the inflated band (configs/components/kj66_compressor_calibration.yaml); the others use the nominal values.

Fixed assumptions: component losses `configs/engine_baseline.yaml` (DePaola values as summarised by the user, off-design loss scaling with corrected flow²); materials and creep settings `configs/materials.yaml` (life 100 h, safety factor 1.5, blade taper At/Ah 1.0 (K = 1.00), pattern factor 0.25; NGV PF_max stored per design).

## Held-out test set (frozen)

`data/splits/phase4_v3_split.csv`: 15 % of design ids, seeded permutation, written before any design was evaluated. Never used for training or tuning.

| split (evaluated designs) | count |
|---|---|
| train_val | 173 |
| test | 27 |

## Design-level outcome

| status | count |
|---|---|
| design_failed | 149 |
| ok | 51 |

Design-mode failure reasons:

| reason | count |
|---|---|
| design_turbine | 149 |

Mechanically feasible (selected material passes creep; tip speed OK): **17** of 51 designed engines; premium material required: 2.

Infeasible designed engines by active constraint(s):

| active constraints | count |
|---|---|
| AN2;TIT_ngv_above_material_data | 26 |
| TIT_ngv_above_material_data | 5 |
| AN2 | 3 |

Material status (designed engines):

| status | count |
|---|---|
| fails_creep | 29 |
| ok_default | 15 |
| unchecked_no_material_data | 5 |
| ok_premium | 2 |

Sensitivity (feasibility recomputed with one setting changed; the defaults above decide feasibility):

| setting | feasible designs |
|---|---|
| at_ah_0.34 | 20 |
| pf_0.15 | 20 |
| pf_0.35 | 13 |

Turbine outside DePaola 2020 Table 2 typical ranges (sanity flags only, not infeasible):

| quantity | designs flagged |
|---|---|
| an2_m2_rpm2 | 7 |
| loading_coefficient | 31 |
| exit_mach | 41 |
| exit_swirl_deg | 2 |

Derived by the sizing (not sampled): n_rpm 25871.7 – 168655; turb_phi 0.247 – 1.962; pr_c 1.686 – 3.193; comp_specific_speed_conventional 0.727 – 1.363

## Point-level outcome (designed engines)

| point | converged | failure reasons | surge flags |
|---|---|---|---|
| cruise_M05_3km | 50/51 | no_convergence: max iterations: 1 | – |
| design | 51/51 | – | – |
| idle_35 | 51/51 | – | – |
| part_speed_80 | 51/51 | – | – |

## Cross-check (pyCycle-CEA, random 2 % of designed engines)

- checked 2 of 51; coupled model vs pyCycle-CEA: flagged 0 (gate passed: True), max |ΔFn| 0.0286 %, max |ΔTSFC| 0.1152 %
- 0-D cycle_simple vs pyCycle-CEA: flagged 0 (gate passed: True)
- secondary metric |ΔFn|/Fg (labelled secondary): max 0.00029

## Cost

Per design (one worker): mean 1.3 s, median 0.09 s, 95th percentile 4.08 s, max 29.33 s.
Last run: 200 designs in 89 s wall on 3 workers.

## Known limitations

- Compressor family is KJ-66-like (one geometry family, one calibration); vaneless diffuser model.
- Turbine efficiency at micro scale unvalidated; incidence-sign convention open (D-029).
- Creep check uses typical (not minimum) rupture data; untapered blades (At/Ah = 1, conservative); MAR-M 247 data stop at 1311 K.
- Turbine geometry outside TurboFlow's recommended ranges (kept, warning in `geom_turb_sizing_warnings`): rotor hub/tip in 0 of 51 designed engines (feasible among them: 0); the loss correlations are less trustworthy there.
- A compressor operating on its choke line can't be resolved (labelled `choke_compressor_*`).
- Design variables are sampled independently in the box: many corners are physically impossible for this engine family (labelled design failures).

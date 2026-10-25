# Dataset card: `phase4_v2` (PILOT)

> **PRELIMINARY DESIGN ONLY - NOT MANUFACTURING-READY. Real micro turbines spin at 100 000+ rpm and need detailed 3-D aerodynamic, structural, rotordynamic, thermal and combustion analysis plus physical testing by qualified people before anything is built or run.**

Status: 200 of 2000 planned designs evaluated (pilot: the first designs of the Sobol sequence).

**STOPPED by the pilot stop rule:** 11 of 200 pilot designs feasible (5.5 % < 10 %). The full run was not started. Analysis: `validation/phase4/pilot_analysis_phase4_v2.yaml`.

Generated with git af26454 by `scripts/phase4/generate_dataset.py` (`turboml/data/generator.py`), config `configs/dataset_phase4_v2.yaml`. Raw records: `datasets/<name>/designs.jsonl` (git-ignored, regenerate from seed + config).

## What one record is

One **design** = cycle design variables + turbine stage choices, sized by the coupled physics model (design mode), checked mechanically (creep, AN², tip speed), then evaluated at 4 **points**: design (SLS), M0.5 at 3 km (design speed), 80 % speed (SLS), idle 35 % speed (SLS). Every design and every point carries `converged` and `failure_reason`; failed designs and points are kept, never dropped.

Physics: 0-D cycle (`cycle_simple` components) with meanline compressor (KJ-66-calibrated, provisional) and meanline axial turbine (TurboFlow formulation); NOT surrogate output. Turbine efficiency at micro scale is unvalidated (flag on every design).

## Design variables (Sobol, scrambled, seed 20261026)

| variable | range | scale | source |
|---|---|---|---|
| pr_c | 2 – 5 | linear | [U] from [R] ('PR < 5' for < 1000 N; Table 3 max 4.7) and JetCat P90 ~2.35 (user). Database: 2.0-4.7 for >= 50 N. |
| W_air_kg_s | 0.08 – 1.9 | log | [U] from [JD] P500 0.90 kg/s @ 500 N, P550 0.93 @ 550 N; 50 N needs ~0.09 and 1000 N ~1.8 at 550-600 N s/kg. Database: e |
| impeller_tip_speed_m_s | 350 – 500 | linear | upper: user decision 2026-10-25 (<= 500 m/s, inside the 400-550 m/s structural range for turbocharger-type impellers, De |
| tit_K | 1000 – 1350 | linear | [U2] User decision 2026-10-25 (pre-Phase 3): raised from 1250 K so AMT-class engines ([R]: EGT up to 875 C = 1148 K, TIT |
| turb_hub_tip | 0.5 – 0.95 | linear | user decision 2026-10-25: rotor-exit hub/tip ratio inside TurboFlow's recommended range [0.5, 0.95] (turboflow geometry_ |
| turb_psi | 0.9 – 2.2 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |
| turb_reaction | 0.2 – 0.6 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |

Compressor calibration factors: 43 of 200 designs sample (Δσ, Cf) uniformly in the inflated band (configs/components/kj66_compressor_calibration.yaml); the others use the nominal values.

Fixed assumptions: component losses `configs/engine_baseline.yaml` (DePaola values as summarised by the user, off-design loss scaling with corrected flow²); materials and creep settings `configs/materials.yaml` (life 100 h, safety factor 1.5, blade taper At/Ah 1.0 (K = 1.00), pattern factor 0.25; NGV PF_max stored per design).

## Held-out test set (frozen)

`data/splits/phase4_v2_split.csv`: 15 % of design ids, seeded permutation, written before any design was evaluated. Never used for training or tuning.

| split (evaluated designs) | count |
|---|---|
| train_val | 166 |
| test | 34 |

## Design-level outcome

| status | count |
|---|---|
| design_failed | 185 |
| ok | 15 |

Design-mode failure reasons:

| reason | count |
|---|---|
| design_compressor_tip_speed_too_low_for_pr | 161 |
| design_turbine | 20 |
| design_compressor_tip_speed_too_high_for_pr | 4 |

Mechanically feasible (selected material passes creep; tip speed OK): **11** of 15 designed engines; premium material required: 2.

Infeasible designed engines by active constraint(s):

| active constraints | count |
|---|---|
| AN2;TIT_ngv_above_material_data | 2 |
| AN2 | 1 |
| TIT_ngv_above_material_data | 1 |

Material status (designed engines):

| status | count |
|---|---|
| ok_default | 9 |
| fails_creep | 3 |
| ok_premium | 2 |
| unchecked_no_material_data | 1 |

Sensitivity (feasibility recomputed with one setting changed; the defaults above decide feasibility):

| setting | feasible designs |
|---|---|
| at_ah_0.34 | 11 |
| pf_0.15 | 11 |
| pf_0.35 | 8 |

Turbine outside DePaola 2020 Table 2 typical ranges (sanity flags only, not infeasible):

| quantity | designs flagged |
|---|---|
| an2_m2_rpm2 | 1 |
| loading_coefficient | 6 |
| exit_mach | 10 |
| exit_swirl_deg | 0 |

Derived by the sizing (option C): n_rpm 28671.8 – 166760; turb_phi 0.268 – 1.327

## Point-level outcome (designed engines)

| point | converged | failure reasons | surge flags |
|---|---|---|---|
| cruise_M05_3km | 14/15 | no_convergence: max iterations: 1 | – |
| design | 15/15 | – | – |
| idle_35 | 15/15 | – | – |
| part_speed_80 | 15/15 | – | – |

## Cross-check (pyCycle-CEA, random 2 % of designed engines)

- checked 1 of 15; coupled model vs pyCycle-CEA: flagged 0 (gate passed: True), max |ΔFn| 0.0104 %, max |ΔTSFC| 0.0539 %
- 0-D cycle_simple vs pyCycle-CEA: flagged 0 (gate passed: True)
- secondary metric |ΔFn|/Fg (labelled secondary): max 0.0001

## Cost

Per design (one worker): mean 6.11 s, median 5.12 s, 95th percentile 10.49 s, max 73.8 s.
Last run: 200 designs in 413 s wall on 3 workers.

## Known limitations

- Compressor family is KJ-66-like (one geometry family, one calibration); vaneless diffuser model.
- Turbine efficiency at micro scale unvalidated; incidence-sign convention open (D-029).
- Creep check uses typical (not minimum) rupture data; untapered blades (At/Ah = 1, conservative); MAR-M 247 data stop at 1311 K.
- Turbine geometry outside TurboFlow's recommended ranges (kept, warning in `geom_turb_sizing_warnings`): rotor hub/tip in 0 of 15 designed engines (feasible among them: 0); the loss correlations are less trustworthy there.
- A compressor operating on its choke line can't be resolved (labelled `choke_compressor_*`).
- Design variables are sampled independently in the box: many corners are physically impossible for this engine family (labelled design failures).

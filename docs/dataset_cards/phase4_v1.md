# Dataset card: `phase4_v1` (PILOT)

> **PRELIMINARY DESIGN ONLY - NOT MANUFACTURING-READY. Real micro turbines spin at 100 000+ rpm and need detailed 3-D aerodynamic, structural, rotordynamic, thermal and combustion analysis plus physical testing by qualified people before anything is built or run.**

Status: 256 of 2000 planned designs evaluated (pilot: the first designs of the Sobol sequence; the full run is started on the user's laptop after the wall-time estimate).

Generated with git e2f2a36 by `scripts/phase4/generate_dataset.py` (`turboml/data/generator.py`), config `configs/dataset_phase4.yaml`. Raw records: `datasets/<name>/designs.jsonl` (git-ignored, regenerate from seed + config).

## What one record is

One **design** = cycle design variables + turbine stage choices, sized by the coupled physics model (design mode), checked mechanically (creep, AN², tip speed), then evaluated at 4 **points**: design (SLS), M0.5 at 3 km (design speed), 80 % speed (SLS), idle 35 % speed (SLS). Every design and every point carries `converged` and `failure_reason`; failed designs and points are kept, never dropped.

Physics: 0-D cycle (`cycle_simple` components) with meanline compressor (KJ-66-calibrated, provisional) and meanline axial turbine (TurboFlow formulation); NOT surrogate output. Turbine efficiency at micro scale is unvalidated (flag on every design).

## Design variables (Sobol, scrambled, seed 20261024)

| variable | range | scale | source |
|---|---|---|---|
| pr_c | 2 – 5 | linear | [U] from [R] ('PR < 5' for < 1000 N; Table 3 max 4.7) and JetCat P90 ~2.35 (user). Database: 2.0-4.7 for >= 50 N. |
| W_air_kg_s | 0.08 – 1.9 | log | [U] from [JD] P500 0.90 kg/s @ 500 N, P550 0.93 @ 550 N; 50 N needs ~0.09 and 1000 N ~1.8 at 550-600 N s/kg. Database: e |
| n_rpm | 50000 – 220000 | linear | [U] P500/P550 80-83 krpm, P220 117 krpm, P80 125 krpm, 50 N engines ~160-200 krpm. Database: >= 50 N engines 61.5-205 kr |
| tit_K | 1000 – 1350 | linear | [U2] User decision 2026-10-25 (pre-Phase 3): raised from 1250 K so AMT-class engines ([R]: EGT up to 875 C = 1148 K, TIT |
| turb_phi | 0.35 – 0.9 | linear | [T] PROVISIONAL: usual axial-turbine stage range (Smith 1965 chart as reproduced in Dixon & Hall, Ch. 4); the Phase 3 ve |
| turb_psi | 0.9 – 2.2 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |
| turb_reaction | 0.2 – 0.6 | linear | [T] PROVISIONAL, as above; verified range of the fast turbine model |

Compressor calibration factors: 53 of 256 designs sample (Δσ, Cf) uniformly in the inflated band (configs/components/kj66_compressor_calibration.yaml); the others use the nominal values.

Fixed assumptions: component losses `configs/engine_baseline.yaml` (DePaola values as summarised by the user, off-design loss scaling with corrected flow²); materials and creep settings `configs/materials.yaml` (life 100 h, safety factor 1.5, taper factor 1.0, pattern factor NOT sourced → NGV PF_max stored).

## Held-out test set (frozen)

`data/splits/phase4_v1_split.csv`: 15 % of design ids, seeded permutation, written before any design was evaluated. Never used for training or tuning.

| split (evaluated designs) | count |
|---|---|
| train_val | 221 |
| test | 35 |

## Design-level outcome

| status | count |
|---|---|
| design_failed | 164 |
| ok | 92 |

Design-mode failure reasons:

| reason | count |
|---|---|
| design_compressor_r2_below_inducer | 51 |
| design_compressor_pr_not_reached_choke_diffuser | 43 |
| design_turbine | 23 |
| design_compressor_pr_not_reached_choke_impeller_exit | 23 |
| design_compressor_choke_impeller_exit | 23 |
| design_compressor_choke_diffuser | 1 |

Mechanically feasible (selected material passes creep; tip speed OK): **3** of 92 designed engines; premium material required: 2.

Infeasible designed engines by active constraint(s):

| active constraints | count |
|---|---|
| AN2;tip_speed | 51 |
| AN2 | 29 |
| AN2;TIT_ngv_above_material_data;tip_speed | 5 |
| AN2;TIT_ngv_above_material_data | 2 |
| tip_speed | 2 |

## Point-level outcome (designed engines)

| point | converged | failure reasons | surge flags |
|---|---|---|---|
| cruise_M05_3km | 90/92 | choke_compressor_inducer: 2 | – |
| design | 92/92 | – | – |
| idle_35 | 92/92 | – | – |
| part_speed_80 | 92/92 | – | – |

## Cross-check (pyCycle-CEA, random 2 % of designed engines)

- checked 2 of 92; coupled model vs pyCycle-CEA: flagged 0 (gate passed: True), max |ΔFn| 0.0386 %, max |ΔTSFC| 0.0669 %
- 0-D cycle_simple vs pyCycle-CEA: flagged 0 (gate passed: True)
- secondary metric |ΔFn|/Fg (labelled secondary): max 0.00039

## Cost

Per design (one worker): mean 2.14 s, median 0.88 s, 95th percentile 6.22 s, max 12.38 s.
Last run: 256 designs in 187 s wall on 3 workers.

## Known limitations

- Compressor family is KJ-66-like (one geometry family, one calibration); vaneless diffuser model.
- Turbine efficiency at micro scale unvalidated; incidence-sign convention open (D-029).
- Creep check uses typical (not minimum) rupture data; blade taper factor 1 (conservative); NGV pattern factor not sourced.
- Turbine geometry outside TurboFlow's recommended ranges (kept, warning in `geom_turb_sizing_warnings`): rotor hub/tip in 61 of 92 designed engines (feasible among them: 0); the loss correlations are less trustworthy there.
- A compressor operating on its choke line can't be resolved (labelled `choke_compressor_*`).
- Design variables are sampled independently in the box: many corners are physically impossible for this engine family (labelled design failures).

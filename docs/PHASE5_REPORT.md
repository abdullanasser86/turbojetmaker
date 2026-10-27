# Phase 5 report: surrogate pipeline (started 2026-10-27; in progress)

User decision (2026-10-27): build the pipeline now on `phase4_v5` and retrain on v6 when it exists. **Every v5
result is "provisional (pre-v6 calibration)"**: v6 changes the turbine losses (k_loss ≈ 1.86, D-064), which moves
design TSFC by ~+19 % on average.

## What exists
| piece | file | notes |
|---|---|---|
| config | `configs/surrogates.yaml` | targets, model settings, OOD thresholds, result labels per dataset |
| data layer | `turboml/surrogates/data.py` | inputs mapped to the sampler's unit cube from the dataset's own config (version-agnostic); calibration factors as extra inputs; targets log/standardized with TRAINING statistics; frozen test ids never used for fitting; seeded validation subset |
| exact GP | `turboml/surrogates/gp.py` | gpytorch, Matern-5/2 ARD, one GP per target; hard limit `max_points_per_output` 10 000 (12 GB VRAM) — error, never silent subsampling |
| deep ensemble | `turboml/surrogates/ensemble.py` | 5 MLPs with Gaussian heads (aleatoric + epistemic), early stopping on validation, checkpoints every 250 epochs (resumable) |
| feasibility classifier | same file | 5-member ensemble, P(feasible) over all sampled designs (design failures = infeasible) |
| metrics | `turboml/surrogates/metrics.py` | RMSE/MAE/R²/relative error; interval coverage, std of standardized residuals, NLL; accuracy/AUC/Brier/ECE + reliability table |
| OOD flags | `turboml/surrogates/ood.py` | outside box; extrapolated tip speed (> 415 m/s, D-060); sparse region (kNN); high ensemble disagreement |
| one command | `scripts/phase5/train_surrogates.py --dataset <name>` | writes `validation/phase5/surrogates_<name>.yaml`, `docs/surrogate_cards/<name>.md`; models in `runs/` |
| tests | `tests/test_surrogates.py` | unit-cube round trip, metrics on a calibrated Gaussian, OOD, GP/ensemble on a smooth function, frozen-split check |

Device-agnostic (`turboml.device.get_device()`): CPU here, CUDA on the RTX 4080 Laptop. v5 on this CPU container:
GP 195 s, ensemble 147 s, classifier 13 s.

## v5 pipeline test — provisional (pre-v6 calibration)
Card: `docs/surrogate_cards/phase4_v5.md`. Frozen test set (300 designs, 159 designed engines):
- Deep ensemble: median relative error 0.02–0.75 % on all 10 targets; 95 % interval coverage 0.94–1.00.
- GP: similar, but over-dispersed for the nearly deterministic targets (mass, diameter: coverage 1.0,
  std(z) 0.3) and under-dispersed for cruise thrust/TSFC (std(z) 1.7–1.9).
- Weak target: cruise TSFC (R² 0.32 for both models). It blows up for engines with almost no net thrust at
  M 0.5 / 3 km (a ratio over a near-zero quantity). For v6, model fuel flow and net thrust separately, or use a bounded
  target.
- Feasibility classifier: accuracy 0.953, ROC AUC 0.976, Brier 0.042, ECE 0.033 (validated tip speed 0.969 / 0.994;
  extrapolated 0.941 / 0.965).
- OOD on the test set: 53.5 % carry the extrapolated-tip-speed flag; 17 % high ensemble disagreement; none outside
  the box or in sparse regions. Surrogate accuracy is the same with or without flags. That is expected: the
  extrapolated flag says the PHYSICS model is extrapolating, not the surrogate. Those designs still need the
  wider-band physics re-check before use.

## Next (after v6)
- `train_surrogates.py --dataset phase4_v6` (one command), then decide the target set (fuel flow / net thrust
  split) and the GP vs ensemble choice per target on validation data only.
- Active learning and multi-objective optimization (qLogNEHVI) on top, with physics verification of every final design.

## v6 retrain (2026-10-27): `phase4_v6`
One command: `uv run python scripts/phase5/train_surrogates.py --dataset phase4_v6`. Card:
`docs/surrogate_cards/phase4_v6.md`; comparison table: `validation/phase5/compare_phase4_v5_vs_phase4_v6.md`
(`scripts/phase5/compare_surrogate_versions.py`).

- New inputs and targets:
  - k_loss (`turbine_loss_multiplier`) is now an input, next to the compressor calibration factors;
  - design and cruise fuel flow are new targets (D-070).
- Run: 1445 train / 255 val / 300 frozen test (757 designed engines in train). CPU timing: GP 174 s,
  ensemble 137 s, classifier 11 s.

**Frozen test set, v5 (provisional) → v6** — each version's own test set, so this compares how well each surrogate
reproduces its own dataset (deep ensemble unless noted):

| target | median rel. error | p95 rel. error | R² | 95 % coverage |
|---|---|---|---|---|
| Fn design | 0.33 → 0.60 % | 2.8 → 5.1 % | 1.000 → 0.999 | 0.99 → 0.97 |
| TSFC design | 0.26 → 0.40 % | 1.7 → 6.8 % | 0.89 → 0.63 | 0.94 → 0.95 |
| **Wf design (new)** | – → **0.36 %** (GP 0.08 %) | – → 1.6 % | – → 1.000 | – → 1.00 |
| Fn cruise | 0.75 → 1.30 % | 7.9 → 34 % | 1.000 → 0.998 | 0.94 → 0.93 |
| TSFC cruise | 0.53 → 0.99 % | 8.4 → 47 % | 0.32 → 0.06 | 0.94 → 0.94 |
| **Wf cruise (new)** | – → **0.58 %** | – → 2.3 % | – → 1.000 | – → 0.99 |
| Fn idle | 0.43 → 0.59 % | 2.7 → 3.8 % | 1.000 → 0.999 | 0.99 → 0.97 |
| T05 idle | 0.21 → 0.30 % | 1.1 → 1.3 % | 0.996 → 0.993 | 0.94 → 0.96 |
| pr_c, T05 design, mass, diameter | ≤ 0.08 % both | ≤ 3.1 % | ≥ 0.97 | 0.99–1.00 |

| feasibility classifier | accuracy | ROC AUC | Brier | ECE |
|---|---|---|---|---|
| v5 (provisional), 93 / 300 feasible | 0.953 | 0.976 | 0.042 | 0.033 |
| v6, 81 / 300 feasible | 0.923 | 0.970 | 0.057 | 0.032 |
| v6, validated tip speed (n 132) | 0.894 | 0.969 | – | 0.042 |
| v6, extrapolated tip speed (n 168) | 0.946 | 0.967 | – | 0.048 |

Reading:
- v6 is a slightly harder dataset: more spread in thrust and SFC because of the k_loss subset and the low-thrust
  designs (D-070). Medians stay below 1 %. The **ratio** targets (TSFC) carry the heavy tail and their R² collapses at
  cruise. **Use the fuel-flow and thrust surrogates and derive TSFC = Wf/Fn**; keep the TSFC surrogates only as
  diagnostics.
- Uncertainty calibration:
  - 95 % coverage 0.93–1.00 for every target;
  - std of standardized residuals ≈ 1 for the thrust/temperature targets;
  - over-dispersed (0.24–0.54) for the near-deterministic geometry targets (mass, diameter, design fuel flow): the
    stated uncertainty there is wider than the error. That is conservative, and harmless for optimisation.
- Classifier:
  - well calibrated (ECE 0.03; reliability bins track the observed frequency within ~0.1, except sparse mid-bins
    with 4–9 designs);
  - the validated-tip-speed subset is the harder one (accuracy 0.89), because the AN² boundary runs through it.
- OOD on the test set (designed engines): extrapolated tip 51 %, high ensemble disagreement 10.5 %, sparse
  0.7 %, outside box 0 %.

## Decisions / next steps (for the user)
- Target set for optimisation: Fn, Wf (→ TSFC), T05, mass, diameter + P(feasible); TSFC surrogates as diagnostics only.
- Model per target: GP (better calibrated on fuel flow, cheap at this size) vs ensemble (smaller tails on thrust).
  Proposal: ensemble for thrust/temperatures, GP for fuel flow and geometry, chosen on validation data only.
- Then Phase 6: multi-objective optimisation (qLogNEHVI) with physics verification of every Pareto design.

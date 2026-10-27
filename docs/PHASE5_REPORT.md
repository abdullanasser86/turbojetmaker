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

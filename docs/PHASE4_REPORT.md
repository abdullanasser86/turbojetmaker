# Phase 4 report: training dataset, pilot (2026-10-25)

> Preliminary design only, not manufacturing-ready. See the scope/safety note in the README.

This phase built the dataset generator, ran a **256-design pilot** here, and stopped before the full 2 000-design run,
as you asked. The full run is cheap (minutes, see the wall-time estimate below). The pilot raises one decision that
should come first: how the design space is sampled.

## Your Phase 3 decisions: what was done
| # | Decision | Done |
|---|---|---|
| 1 | Incidence sign selectable; both validation runs recorded; issue text drafted, not posted | `turbine_design.incidence_sign` (`turboflow` default / `rotor_plus`); D-036; `docs/upstream_issues.md` (5 drafts: incidence sign, Wiesner slip degrees→radians, `print_simulation_summary` TypeError, lossless algebraic diffuser, 1-D diffuser TypeError; the last two still need stand-alone reproducers) |
| 2 | Creep from Nickel Institute No. 393; life and SF in config; rotor at relative T; NGV at TIT + PF(TIT − T3); infeasible only if the selected material fails; active constraint reported | `turboml/physics/creep.py`, `validation/cases/ni393_cast_rupture.csv` (Tables 9, 10 pp. 14–15; densities Table 1 p. 5); D-037. **Pattern factor not sourced** (see open issues): PF_max stored instead |
| 3 | 20 % calibration sampling; widen band until ≥ 95 % held-out coverage | k = 2.75 × the 2σ half-widths → 34/34 held-out points inside; D-038 |
| 4 | DePaola nozzle/inlet = total-pressure ratios; loss ∝ corrected flow² with a citation | Config sources updated; Walsh & Fletcher (2nd ed. 2004), chapter-level, page to check; D-039 |
| 5 | OS / CUDA index | **Not done**: OS field still the placeholder, and download.pytorch.org is blocked from this container (see decisions) |
| 6 | Documents | **Only Nickel No. 393 arrived** (the second PDF was an unrelated *Energies* paper). DePaola, Murray, Xiang, JetCat and Benner did not arrive, so calibration and turbine checks were not re-run against them |
| 7 | Phase 4 scope: 4 points, 2 000 designs, failure reasons, frozen 15 % test set, laptop-friendly, wall-time report, profiling + cheap speedups only | All done; pilot of 256; full run waiting for you |
| 8 | Idle EGT / micro-scale turbine efficiency tracked | Still open (no Murray data). Turbine-efficiency flag on every design card and record |

## What was built
- **Creep checks** (`turboml/physics/creep.py`, `mechanical.py`):
  - Rupture strength of IN-713C (default) and MAR-M 247 (premium) at 100 h / 1000 h. Log-stress interpolation in
    temperature; Larson–Miller only for other lives; **no extrapolation above the data**.
  - Rotor: centrifugal (AN²) + gas-bending stress at the rotor-relative total temperature.
  - NGVs: gas bending at TIT + PF (TIT − T3).
  - Safety factor 1.5, life 100 h.
  - The failing check is reported as the active constraint (AN2, creep_rotor, creep_ngv,
    TIT_*_above_material_data, tip_speed). The design card shows the full creep table.
- **Profiling and speedup** (D-040, `validation/phase4/profiling.yaml`):
  - 90 % of the time was the vaneless-diffuser continuity solve. It is now a warm-started Newton step with the old
    solver kept as fallback.
  - Engine evaluations are **3.3–4.3× faster**, outputs identical to 3e-11.
  - Next hot spot: the gas-property inversion `thermo.T_from_h` (~39 %). Left alone; not needed.
- **Robust sizing and continuation** (D-041): a bracketed compressor sizing search, and adaptive speed and inlet-state
  continuation. Design failures now mean "no geometry exists", not "the solver started badly".
- **Generator** (`turboml/data/generator.py`, `scripts/phase4/`):
  - Scrambled Sobol, 7 variables, 4 points per design.
  - Frozen split `data/splits/phase4_v1_split.csv` (300 test ids of 2 000), written before any evaluation; the script
    refuses to change it.
  - Append-only JSONL with fsync per design; restart skips finished ids; a line cut off by a crash is repaired.
  - joblib workers = physical cores − 1.
  - Run log in `runs/`; flattened `designs.csv` / `points.csv`.
  - Scripts: pyCycle-CEA cross-check, dataset card, wall-time estimator.
- **Notebook** `notebooks/04_dataset.ipynb` (executed). Notebook 03's materials cell was updated to the creep check
  (the old temperature-limit function no longer exists).

## Dataset card and failure statistics (pilot, 256 designs)
Full card: `docs/dataset_cards/phase4_v1.md`; numbers: `validation/phase4/dataset_stats_phase4_v1.yaml`. Generated at
commit e2f2a36; a second run reproduced every status, flag and thrust bit for bit.

| Outcome | Count |
|---|---|
| designs evaluated | 256 (train/val 221, test 35) |
| design mode failed | **164 (64 %)** |
| — compressor: PR reached only with r2 < inducer tip (`r2_below_inducer`) | 51 |
| — compressor: PR not reached before diffuser choke | 43 |
| — compressor: PR not reached before impeller-exit choke | 23 |
| — compressor: chokes at its own design point (impeller exit) | 23 |
| — turbine sizing failed | 23 |
| — compressor: chokes at its own design point (diffuser) | 1 |
| designed engines | 92 |
| **mechanically feasible** | **3** (1 needs MAR-M 247) |
| infeasible, active constraints: AN2 + tip_speed | 51 |
| — AN2 | 29 |
| — AN2 + NGV above material data (+ tip_speed) | 7 |
| — tip_speed only | 2 |
| off-design convergence (designed engines): M0.5/3 km | 90/92 (2 × `choke_compressor_inducer`: the design point is within ~2 % of inducer choke) |
| — 80 % speed | 92/92 |
| — idle 35 % | 92/92 |
| surge flags | 0 |
| calibration-sampled designs | 53 (21 %) |
| cross-check vs pyCycle-CEA (2 of 92) | 0 flagged; max ΔFn 0.039 %, ΔTSFC 0.067 %; gate passed |
| NGV PF_max (IN-713C) | p10 0.04, median 0.24, p90 0.52 |

What the pilot says:
- **The turbine blade root (AN²) is the binding constraint.** Across designed engines, 1.5 × root stress / IN-713C
  allowable has a median of 3.2 (range 0.7–10.7). Gas bending is only ~2 % of the centrifugal stress.
- **Long turbine blades drive it.**
  - 61 of 92 designed engines have a rotor hub/tip ratio below TurboFlow's recommended range [0.5, 0.95], down to 0.01.
  - None of those 61 is feasible; all 3 feasible designs have hub/tip 0.63–0.71.
  - Turbine size comes from the sampled flow and loading coefficients (φ, ψ) at the sampled shaft speed, so many
    combinations give an unrealistic turbine.
- **Tip speed:** 58/92 exceed the 500 m/s impeller limit (configs/bounds.yaml). Tip speed correlates with PR (r = 0.94):
  PR > ~3 at this compressor's work coefficient needs > 500 m/s.
- The 3 feasible designs: 47–92 N, PR 2.2–2.7, TIT 1004–1145 K, 78–120 krpm.

## Wall time of the full run (2 000 designs)
Measured here: 256 designs in 187 s on 3 workers (Xeon 2.1 GHz). The mean is 2.2 s per design per worker (median 0.9,
p95 6.1, max 12 s), so the full run is **≈ 4 300 worker-seconds**.

| Machine | Workers | Expected wall time |
|---|---|---|
| this container | 3 | ≈ 24 min |
| 8-core laptop | 7 | ≈ 10 min at this container's per-core speed, ≈ 5–7 min on a modern laptop core |
| 16-core laptop | 15 | ≈ 5 min / ≈ 3–4 min |

I don't know your CPU (the OS/hardware field is still the placeholder). To measure it on your machine without
writing data, plugged in and in performance mode (~1–2 min):
```bash
uv run python scripts/phase4/estimate_walltime.py
```
The full run itself: `uv run python scripts/phase4/generate_dataset.py`. It is resumable: just re-run it after an
interruption. Then run `crosscheck_dataset.py` and `make_dataset_card.py`.

## Tests
`uv run pytest`: **94 passed** (2 min 25 s), including 11 mechanical/creep tests and 6 generator tests (sampling
reproducibility, calibration sampling, frozen split equals the committed file, failures recorded, all four points on a
baseline-like design, resume helpers). Ruff, black and `uv lock --check`
are clean.

## Open issues
1. **Combustor pattern factor not sourced.** I could not find a verifiable cited range (DePaola Sec. 2.5.2 and
   Lefebvre & Ballal were not available here). The config stays `null`; every design stores PF_max, so feasibility can
   be recomputed from the column without re-running. Designs failing at PF = 0 are already infeasible.
2. **MAR-M 247 data stop at 1900 °F (1311 K), lower than IN-713C (2000 °F, 1366 K).** Above 1311 K the "premium" alloy
   cannot be checked at all. Higher-temperature MAR-M 247 rupture data would be needed for the 1311–1350 K TIT range.
3. **Blade taper factor K = 1** (constant-section blade) is conservative. Real turbine blades taper, which lowers
   root stress. What-if on the pilot: K = 0.8 → 7 feasible, 0.67 → 9, 0.5 → 11 (of 92). No sourced value yet.
4. **Idle EGT** 654 °C at 35 % speed vs JetCat's ~480 °C minimum. Not tuned. The re-check against Murray's measured KJ-66
   data is not possible without the thesis. Next step if the gap persists: a cited small-turbine Reynolds-number
   efficiency correction and the idle combustor efficiency.
5. **Turbine efficiency at micro scale unvalidated**: flagged on every design and design card. The hub/tip finding
   above makes it worse: most designs are outside TurboFlow's correlation range.
6. **Calibration band**: the held-out points were used to size the band, so they are no longer independent evidence.
7. Cross-check sample: 2 % of the pilot is only 2 engines. The full run gives ~14.

## Decisions I need from you
1. **Sampling parametrization before the full run** (the main one). With the box as approved, ~64 % of runs fail in
   design mode and ~1 in 30 designed engines is feasible, so 2 000 designs give ≈ 25 feasible engines.
   - **A. Run as configured.** Simple; the surrogate learns the feasibility boundary well, but the region the optimizer
     cares about has very few examples.
   - **B. Tie shaft speed to the thrust trend** (n = database trend × U(0.75, 1.3)). Tested on 96 designs: design
     failures 35 % instead of 64 %, but feasible still ~1/62. Doesn't address the turbine.
   - **C. Sample variables that map onto the constraints** (recommended): impeller tip speed (≤ 500 m/s) instead of
     shaft speed, and rotor hub/tip ratio in TurboFlow's range [0.5, 0.95] instead of φ. Shaft speed and φ then follow
     from sizing. It costs ~half a day of code and a new dataset version (`phase4_v2`, new frozen split). The surrogate
     sees fewer impossible engines and more feasible ones.
2. **Blade taper factor**: keep K = 1 (conservative), or give me a source for a typical value?
3. **Pattern factor**: a source (DePaola Sec. 2.5.2, Lefebvre & Ballal, or another reference) would turn the NGV check
   on.
4. **OS and PyTorch:**
   - On **Linux**, the default PyPI torch wheels already include CUDA 12.x; nothing to change.
   - On **Windows**, the lock needs the PyTorch CUDA 12.x index, and download.pytorch.org is blocked by this
     environment's network policy. Allowing that host in the cloud environment's Network access settings
     (https://code.claude.com/docs/en/cloud-environments#network-access) lets me add it to `uv.lock` and test it in
     `check_env.py`.
5. **Documents** (not received): DePaola 2020, Murray 2009, Xiang 2015 App. A–I, JetCat spec pages, Benner 1997.
6. **Upstream issues:** the drafts are in `docs/upstream_issues.md`; posting them is your call.
7. "Go" for the full run (after decision 1), then Phase 5.

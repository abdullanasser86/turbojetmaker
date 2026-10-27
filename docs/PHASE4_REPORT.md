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

---

# Phase 4, round 2 (2026-10-25): reference data, KJ-66 rebuild, idle EGT, option-C pilot

## What was done, item by item
| # | Your instruction | Done |
|---|---|---|
| 0 | Ingest the extracted reference data, tagged MEAS/CFD/MFR/ASSUM, weight MEAS > CFD > MFR > ASSUM | Four new files in `validation/cases/` plus citations in `configs/` (D-044). The 10 % same-model spread is recorded as the error floor for MFR comparisons |
| A | Rebuild the KJ-66 compressor from Xiang's geometry | Same geometry was already in use, so **no calibration parameter is removed**: Δσ still stands in for the unpublished blade angles/backsweep, Cf for the unpublished 15-vane diffuser. Recalibration reproduces Δσ −0.1041 / Cf 0.0937 (D-047) |
| B | Idle-EGT investigation in your order | Done, no tuning; findings below (D-048) |
| C | DePaola assumptions, Table 2 ranges, eq. 58 | Component values unchanged (now tagged). Table 2 ranges are sanity flags on every design. Taper expressed as At/Ah, default 1 (D-045) |
| D | JetCat specs | `validation/cases/jetcat_specs_user_2026.csv`. Matches the committed catalogue within ≤ 2 % except P220 max EGT 720 vs 750 °C |
| 1 | Option C sampling, new version, 200-design pilot, stop rule | `phase4_v2` with its own frozen split. **Pilot 5.5 % feasible → stopped**; analysis below (D-046, D-049) |
| 2 | Taper At/Ah = 1 default, 0.34 as a sensitivity column | Done; counts below |
| 3 | Pattern factor 0.25 (0.15–0.35) | Done; PF_max still stored |
| 4 | MAR-M 247 gap → "unchecked, infeasible (no material data)" | `material_status = unchecked_no_material_data`; counts below |
| 5 | OS / PyTorch index | OS field is still the placeholder. download.pytorch.org is now reachable (HTTP 200), so I can add the CUDA 12.x index as soon as you confirm Windows |
| 6 | Upstream issues stay drafts | Unchanged in `docs/upstream_issues.md` |

## Calibration changes
**None.** The recalibration with the current code reproduces the stored factors exactly. Model vs reference data
(`validation/phase4/kj66_rebuild_check.yaml`):
- **MEAS** (Murray, 70–110 krpm): model peak efficiency 0.754–0.768 vs measured 0.75–0.77. PR at peak efficiency
  within −2.2 … +1.1 % of the measured maxima.
- **CFD** (Xiang, 80/100/117 krpm): the CFD's efficiency drop 0.73 → 0.63 → 0.55 is **not** reproduced; the model
  stays at 0.76–0.77. The measurements stay at 0.77 → 0.75, so under MEAS > CFD the model is kept. Xiang's own text
  says the CFD under-predicts the measured PR at speed.
- **MFR**: PR 2.2 at 0.22 kg/s and 117 krpm → model 2.34 (+6.4 %). PR ≈ 2.54 at 128 krpm (Murray) → model 2.67
  (+5 %, extrapolated). Both are inside the 10 % same-model spread.

## Idle-EGT findings
1. **The original comparison mixed engines.** The 654 °C idle EGT belongs to our ~150 N baseline (TIT 1150 K, EGT
   759 °C at full power), not to a KJ-66.
2. **KJ-66 replicas at full power are close.** Compared as a thermocouple reading:
   - Murray build, matched to 92.3 N and SFC 0.154: 576 °C vs 590 °C spec (−14 K).
   - Xiang build, matched to 75 N at 0.22 kg/s: 635 °C vs 570 °C (+65 K).
3. **But the model's EGT hardly falls towards idle.** On the replicas, idle EGT (28–35 % speed) is 574–620 °C, about
   the same as full power. JetCat's idle-to-max EGT bands are 190–330 K wide. That is the real symptom: the trend's
   shape, not one number.
4. **Your four suspects, in order:**
   1. Idle efficiencies (η_c 0.70, η_t 0.72): each point of efficiency is worth only ~10 K, so both would need about
      +10 pts.
   2. Static vs total EGT: ≤ 1 K at idle (6–9 K at full power).
   3. Combustion efficiency at idle 0.90 / 0.80: −1 / −4 K. In a speed-controlled, power-balanced engine the turbine
      sets the TIT; η_b only changes the fuel flow (+6 / +19 %).
   4. Reynolds number at idle is ~0.27 of design. A correction lowers efficiencies, so idle would get *hotter*: wrong
      direction.
5. **Next suspect: the low-speed matching.** Idle sits at 28–35 % speed, well below the 70 krpm compressor calibration
   data; the turbine and nozzle flow capacity at PR ≈ 1.1 also matters. Murray's measured 48–64 krpm PR, mass flow and
   TIT (Figs 5.1–5.3) would test this directly.

## Option-C pilot (`phase4_v2`, 200 designs): STOPPED (5.5 % < 10 %)
Dataset card: `docs/dataset_cards/phase4_v2.md`. Analysis: `validation/phase4/pilot_analysis_phase4_v2.yaml`.

| | count |
|---|---|
| design mode failed | 185 |
| — compressor: tip speed too low for the PR | 161 |
| — turbine sizing (limit loading, no convergence, axial Mach ≥ 1 at high hub/tip) | 20 |
| — compressor: tip speed too high for the PR | 4 |
| designed engines | 15 |
| **feasible** | **11** (5.5 % of the pilot, 73 % of designed engines) |
| infeasible, active constraints | AN2 + TIT_ngv_above_material_data: 2; AN2: 1; TIT_ngv_above_material_data: 1 |
| material status | ok_default 9, ok_premium 2, fails_creep 3, **unchecked_no_material_data 1** (TIT 1274 K) |
| sensitivity: feasible at At/Ah 0.34 / PF 0.15 / PF 0.35 | 11 / 11 / 8 |
| off-design convergence (designed) | design 15/15, cruise 14/15 (1 no_convergence), 80 % 15/15, idle 15/15 |
| DePaola Table 2 flags | exit Mach 10, ψ 6, AN² 1, exit swirl 0 |
| cross-check vs pyCycle-CEA | 1 checked, 0 flagged (ΔFn 0.010 %) |

**Are the conservative settings the cause? No.** Feasibility recomputed from the stored stresses and temperatures:

| setting | feasible / 200 |
|---|---|
| default (100 h, SF 1.5, At/Ah 1, PF 0.25) | 11 |
| life 50 h / SF 1.25 / SF 1.0 / At/Ah 0.34 / PF 0.15 (one at a time) | 11 each |
| all relaxed together | 13 (6.5 %) |

**The cause is the PR–tip-speed coupling of this compressor family.**
- With a fixed KJ-66-like wheel shape, the pressure ratio is set mainly by the tip speed. Better efficiency at higher
  specific speed buys only ~50 m/s of range at a given PR.
- No design above PR ~3.2 can be built at ≤ 500 m/s; 60 % of the PR box (3–5) is unreachable.
- Even for PR 2–3, an independently drawn tip speed misses the narrow reachable band about half the time.
- The MAR-M 247 gap affected 1 designed engine. Capping that branch at 1311 K would change nothing in this pilot, so I
  don't propose it yet.

## Decisions I need from you
1. **How to sample before the full run** (feasibility-guided proposals, as you asked):
   - **A. Make PR a result, not an input (recommended).** Sample tip speed U2 (350–500 m/s), the compressor's
     specific speed (or shaft speed), W, TIT, turbine hub/tip, ψ and R. The compressor is then sized directly, with no
     root search, and PR comes out wherever this family puts it (roughly 1.9–3.3). Every sample is a real wheel, the tip-speed
     limit holds by construction, and it is faster. The PR range is what this family can reach, which is the honest
     answer. The specific-speed bounds would be derived from a sizing scan, like the U2 lower bound.
   - **B. Keep (PR, U2) but cap PR at ~3.2.** Minimal change, but the narrow U2 band still wastes ~half the samples;
     I'd expect roughly 15–20 % feasible.
   - **C. Sample PR within its reachable band at each U2.** This needs two extra sizings per design to find the band
     edges (about +10 s per design).
   - **D. Widen the compressor family** (backsweep / work coefficient as a variable) to reach PR 3–5 below 500 m/s.
     This needs blade-angle data we don't have; a longer-term option.
2. **Murray's low-speed data:** images of Figs 5.1–5.3 / App. D pages, so I can digitize the 48–64 krpm points
   (labelled digitized). This is the most direct test of the idle-EGT problem.
3. **OS** (Windows 11 or Linux) for the PyTorch CUDA index.
4. "Go" for the chosen sampling, then the full run.

---

# Phase 4, round 3 (2026-10-25): Murray validation, option-A pilot, Windows

## What was done, item by item
| # | Your instruction | Done |
|---|---|---|
| 1 | Option A: sample tip speed + compressor specific speed, PR as output; every sample buildable; new version + frozen split; pilot, then full run if ≥ 10 % | `engine.size_compressor_direct`. Variable n_s,U = ω√Q01/U2^1.5 in 0.35–0.90; U2 350–500 m/s. Every corner of the box builds (test). Dataset `phase4_v3` with its own split. **Pilot 8.5 % feasible → stopped; no full run** (D-053, D-054) |
| 1 | Record the PR range and the excluded high-PR engines; option D as an open item | D-053. PR lands at 1.7–3.35 with these bounds (your wording: ~1.9–3.3). It brackets KJ-66 (2.2–2.54) and JetCat P90 (≈ 2.35). PR 3.4–5 engines are excluded until a wider compressor family (option D) has blade-angle data |
| 2 | Digitize Murray, label digitized [MEAS] with uncertainty, store in `validation/cases/murray2009/` | **Your upload holds thesis pp. 121–171, so Figs 5.1–5.3 and the maps 5.12–5.19 were not in it.** Used instead (D-051): Appendix D's 15 running-data tables (*transcribed*, more accurate than digitizing) and the Appendix C run plots (*digitized*, ±12 °C / ±0.8 kPa at 2σ, measured against Appendix D), including the three turbojet runs at 28–120 krpm |
| 2 | Second validation case for compressor and turbine, calibration/validation kept apart | Compressor: 13 validation points + 2 near-calibration. Turbine: indicative only (geometry unknown) (D-052) |
| 2 | Redo the idle-EGT study against the low-speed points, recovery 0.68 | Done with the turbojet runs; findings below |
| 3 | Windows 11: CUDA index in the lock, check_env, PowerShell README, spawn-safe, test the generator with workers | Done (D-050). Smoke test passes here (Linux, spawn pool + loky + the real generator); **not yet run on Windows itself** |
| 4 | Upstream issues stay drafts | Unchanged |

## Dataset card and feasibility / failure statistics: `phase4_v3` pilot (200 designs)
Card: `docs/dataset_cards/phase4_v3.md`. Analysis: `validation/phase4/pilot_analysis_phase4_v3.yaml`.

| | count |
|---|---|
| design failed | 149: **all in turbine sizing** (85 hub/tip too high for the flow (axial Mach ≥ 1), 30 limit loading, 34 no convergence) |
| compressor failures | **0** (option A works) |
| designed engines | 51 (PR 1.69–3.19, 26–169 krpm) |
| **feasible** | **17 (8.5 % of the pilot, 33 % of designed engines)** |
| infeasible, active constraints | AN² + NGV above material data: 26; NGV above material data: 5; AN²: 3 |
| material status | ok 713C 15, ok MAR-M 247 2, fails creep 29, **no material data 5** (TIT 1134–1213 K) |
| sensitivity: At/Ah 0.34 / PF 0.15 / PF 0.35 | 20 / 20 / 13 feasible |
| off-design convergence (designed) | design 51/51, cruise 50/51, 80 % 51/51, idle 51/51 |
| cross-check vs pyCycle-CEA | 2 checked, 0 flagged (ΔFn ≤ 0.03 %, ΔTSFC ≤ 0.12 %) |
| time | 200 designs in 89 s on 3 workers |

What limits feasibility now:
1. **Turbine hub/tip.** Option A gives lower-work cycles, and with ψ fixed a high hub/tip ratio needs a supersonic
   axial velocity. Build success by hub/tip: 0.5–0.6: 50 %, 0.6–0.7: 45 %, 0.7–0.8: 18 %, 0.8–0.95: 1 %.
2. **NGV material data.** With PF 0.25 the hottest vane exceeds IN-713C's rupture data (1366 K) above TIT ≈ 1150–1190
   K. No design above TIT 1145 K is feasible.
3. **Conservative settings are a minor factor.** Relaxing all together gives 33 feasible (16.5 %); one setting at a
   time gives 17–21.

## Murray validation results
- **Compressor** (validation points):
  - PR rms 2.1 % (max 5.2 % at 120 krpm), −1.1 … −1.6 % at 48–64 krpm.
  - Efficiency rms 3.0 pts; the model is nearly flat at 0.75–0.76, the measurement drops at 120 and 48 krpm.
  - The 88/91 krpm near-calibration points: 0.8 % / 1.6 pts.
  - Plot: `validation/phase4/murray2009_compressor.png`.
- **Turbine** (indicative): flow capacity +5.5 %, efficiency **+10 … +13 pts** above Murray's derived values. This is
  the first measured sign that the turbine-efficiency flag matters, roughly 10 pts optimistic at KJ-66 size.
- **Turbojet engine** (TIT fitted at 120 krpm only): EGT +18 K at 104–120 krpm, +36 K at 62–96, +53 K at 48–56 and
  +41 K at 28–32 krpm (ψ 1.5). Part-speed air flow is −8.5 % and delivery pressure −5 kPa.

## Idle-EGT findings (redone with measured low-speed data)
1. **The real KJ-66 is hot at idle.** Thermocouple readings in the turbojet runs:
   - 28–32 krpm: 532–538 °C;
   - 48 krpm: 485–527 °C;
   - 120 krpm: 536–550 °C.

   EGT is roughly flat with a slight U shape, so the round-2 assumption (idle much cooler than full power, from
   JetCat bands) does not hold for this engine.
2. **Thermocouple:** Murray already corrects T02 and T05 to total with r = 0.68. Our comparison uses the raw reading
   on both sides; the correction is 2–3 K at idle and ~15 K at 120 krpm.
3. **Our model is ~40–80 K too hot at low speed** and starts its low-speed rise too early (below ~80 krpm instead of
   ~48 krpm). Its compressor speed lines are right to ~2 %, but its operating line carries 8–9 % less air, so the gap
   is in the turbine/nozzle flow matching at low pressure ratio. The indicative turbine check (+10–13 pts efficiency,
   +6 % flow capacity) points the same way.
4. Combustion efficiency (Murray measured 69–77 %, ±5–22 %) and the Reynolds effect remain as in round 2: neither
   explains a hot idle.
5. Next step, if you want this closed: KJ-66 turbine geometry (blade count, radii, angles) or Murray's turbine map
   figures (5.16–5.19) for a real turbine validation.

## Windows environment check
- `uv.lock`: torch 2.10.0+cu128 from download.pytorch.org on Windows; 2.10.0 from PyPI elsewhere.
- `scripts/check_env.py` on Windows also fails on:
  - a CPU-only torch;
  - CUDA not available;
  - UTF-8 mode off.
- README: PowerShell setup (uv, `setx PYTHONUTF8 1`, `uv sync --frozen`, check_env, smoke test).
- Text I/O is UTF-8 everywhere, and all entry points are guarded.
- `scripts/phase4/windows_smoke_test.py`: PASS here. Please run it once on the laptop:
  `uv run python scripts/phase4/windows_smoke_test.py`.

## Decisions I need from you
1. **Turbine side of the sampling** (the same "buildable by construction" fix as option A, now for the turbine):
   - **T1 (recommended):** sample the hub/tip ratio *within its reachable range*. 0.5 up to the hub/tip at which the
     rotor-exit axial Mach number reaches DePaola's exit-Mach limit of 0.5 (Table 2), computed analytically per design
     (cheap). Every turbine is then buildable and inside DePaola's range.
   - **T2:** sample the flow coefficient φ (0.35–0.9, the model's verified range) and derive hub/tip, as in v1.
     Buildable, but v1 showed many long-bladed turbines (hub/tip < 0.5) that then fail AN².
   - **T3:** narrow hub/tip to 0.5–0.75 (data-driven). Simple, but ~55 % would still fail.

   With T1 I expect roughly 25–35 % feasible (about all built × the 33 % feasible-when-built), so above your 10 %.
2. **TIT above ~1150 K** is not shown feasible with PF 0.25 and the IN-713C data. Keep it sampled, so the
   feasibility classifier learns the edge (recommended), or cap TIT? Higher-temperature rupture data (MAR-M 247 above
   1900 °F, or another cast alloy) would open it.
3. **Murray Ch. 5 pages** (Figs 5.1–5.3, 5.12–5.19) if you want the measured maps as well; the Appendix D tables
   already give the operating points.
4. "Go" for the chosen turbine option → new pilot (`phase4_v4`) → full run if ≥ 10 %.

---

# Phase 4, round 4 (2026-10-25): T1 turbine sampling, high-TIT share, Murray Chapter 5, full v4 dataset

## What was done, item by item
| # | Your instruction | Done |
|---|---|---|
| 1 | T1: hub/tip sampled only inside its buildable range, upper limit at turbine exit Mach 0.5; range computed per design before sampling; derivation in DECISIONS; report the sampled range and how often Mach 0.5 binds | `turbine_design.max_hub_tip_for_exit_mach`, D-055. Sampled hub/tip **0.50–0.84**. The Mach-0.5 limit was the binding upper bound for **1090/1090** designed engines (it is always below 0.95). **905/2000** designs have no buildable hub/tip (exit Mach > 0.5 even at hub/tip 0.5; high ψ → small mean radius) and are labelled |
| 2 | Keep TIT > 1150 K at ~15 %; feasibility inside/outside the band; flag infeasible high-TIT designs by active constraint; open item for higher-temperature rupture data | Piecewise Sobol map → exactly 15.0 % (300/2000), D-056. Split and constraints below; open item recorded |
| 3 | Digitize Ch. 5 into `validation/cases/murray2009/` (digitized [MEAS], uncertainty, published bands); validate compressor and turbine maps with calibration/validation separated; PR, efficiency and corrected-flow errors per speed line; idle check against Fig 5.3; tell before recalibrating | Done (D-057, D-058). **The maps contradict the calibration at 110–120 krpm. Not refitted; decision below** |
| 4 | v4 pilot → full run if ≥ 10 % | Pilot **32 %** feasible → full 2000 run done (964 s on 3 workers) |

## Dataset card and feasibility / failure statistics: `phase4_v4` (2000 designs, full run)
Card: `docs/dataset_cards/phase4_v4.md`. Pilot analysis: `validation/phase4/pilot_analysis_phase4_v4.yaml`.

| | count |
|---|---|
| designs | 2000 (train/val 1700, test 300, frozen before evaluation) |
| design failed | 910: no buildable hub/tip 905, other turbine sizing 5; **compressor 0** |
| designed engines | 1090 (PR 1.71–3.29, 21–268 krpm, design thrust 16–804 N among the feasible) |
| **feasible** | **625 (31.3 % of all designs, 57 % of designed engines)**; test split: 92 |
| infeasible designed engines | AN²: 213; AN² + NGV above material data: 194; NGV above material data: 47; rotor creep (± NGV): 11 |
| material status (designed) | IN-713C 509, MAR-M 247 116, fails creep 418, no material data 47 |
| sensitivity (feasible) | At/Ah 0.34: 874; PF 0.15: 668; PF 0.35: 485 |
| off-design convergence (designed) | design, M0.5/3 km, 80 %, idle: **1090/1090 each** |
| cross-check vs pyCycle-CEA | 22 checked (2 %), **0 flagged** (max ΔFn 0.026 %, ΔTSFC 0.10 %) |

**High-TIT split** (TIT > 1150 K = 15 % of samples):
| band | designs | designed | feasible | % of designs |
|---|---|---|---|---|
| TIT > 1150 K | 300 | 160 | **3** | 1.0 % |
| TIT ≤ 1150 K | 1700 | 930 | 622 | 36.6 % |

Infeasible high-TIT designs by active constraint (material status):
- AN² + NGV above material data (fails creep): 139;
- NGV above material data only (unchecked, no material data): 16;
- rotor creep + NGV above data: 2.

So every infeasible high-TIT design hits the missing high-temperature rupture data (open item, D-056); most also fail
AN².

## Murray validation results
- **Compressor maps (Figs 5.12/5.14, stored calibration):**
  - **50–100 krpm:** PR −1.2 … +2.0 % (within the published uncertainty), corrected-flow offset −3.1 … +0.6 %,
    efficiency ±3 pts rms.
  - **110 krpm:** PR +4.1 %; **120 krpm:** PR +7.1 % and efficiency +6.1 pts. Both are outside the published
    uncertainty on every point (`validation/phase4/murray2009_maps.png`).
- **Turbine maps (Figs 5.16/5.18, indicative replica, geometry unknown):** the flow-capacity shape and choke are
  reproduced (+1 % at 68 krpm, growing to +18 % at the lowest PR). Efficiency is +10 … +12 pts above Murray's derived
  values on every speed line.
- **Operating lines (Figs 5.1/5.2):** the turbojet replicas lie within −0.015 PR and −5 % corrected flow of the
  all-run envelope.

## Idle-EGT check against Fig 5.3 and the low-speed points
- Murray's turbojet TIT (calculated by him from the exhaust temperature and the compressor work) is 780–820 K at 48
  krpm and 920–940 K at 119 krpm.
- Our replicas (TIT fitted at 120 krpm only) stay at 840–860 K below ~70 krpm, so they are **25–65 K too hot at low
  speed** (ψ 1.5: +39 K mean below 60 krpm).
- That matches the low-speed EGT over-prediction found with the run plots (+40–80 K, thermocouple reading with
  r = 0.68).
- The extra heat is in the matched TIT, and the compressor is right there (±1–2 %). So the turbine/nozzle behaviour at
  low pressure ratio is the prime suspect; the indicative turbine map points the same way. Not tuned.

## Decisions I need from you
1. **Compressor calibration at 110–120 krpm** (you asked to be told before any recalibration). The measured maps show
   the model ~4–7 % optimistic in PR above ~110 krpm (U2 ≳ 380 m/s). This matters for the dataset: **54 % of the
   designed v4 engines (42 % of the feasible ones) have U2 > 415 m/s**, beyond the measured 120 krpm line.
   - **R1 (recommended):** refit Δσ and Cf on several measured lines (e.g. 70, 90, 110 krpm), keeping 50/60/80/100/120
     for validation, then regenerate v4 as v5.
   - **R2:** keep the calibration, add a tip-speed / inducer-Mach loss term (a model change that needs a source).
   - **R3:** keep everything and flag designs above U2 ≈ 415 m/s as "compressor beyond validated range" (cheapest; the
     surrogate learns an optimistic compressor there).
2. **Turbine at low PR / idle**: the model is 25–65 K hot at idle. Do you want this pursued now (it needs the KJ-66
   turbine geometry: blade count, radii, blade angles) or tracked as an open item?
3. **Higher-temperature rupture data** (open item): any source for MAR-M 247 above 1900 °F, or another cast alloy,
   would open TIT > 1150 K, which currently has 1 % feasible.
4. "Go" for Phase 5 (surrogates) on `phase4_v4`, or after R1 on a v5.

# Phase 4, round 5 (2026-10-27): compressor refit, KJ-66 turbine calibration, Nickel 337, `phase4_v5`

## What was done, item by item
| # | Your instruction | Done |
|---|---|---|
| 1a | R1: refit on several of Murray's measured speed lines, hold the rest out, report held-out PR / η / flow errors per line | Calibration 50/70/90/110 krpm, held out 40/60/80/100/120 (D-059). Table below |
| 1b | R3: flag tip speed > 415 m/s "extrapolated", widen its band, justify by the error growth toward the top line | Quadratic growth law of the refit PR error vs U2; W(500 m/s) = 2.67, one-sided toward the measured bias (D-060). Every design carries `tip_speed_regime` |
| 1c | v5 with a fresh frozen test set; feasible fraction inside / outside the validated tip speed | Seed 20261103, split frozen before evaluation (300 test ids). Table below |
| — | No R2 term without a published source; propose first | Not added. Proposal below |
| 2 | KJ-66 turbine: 66 mm / 18 NGV / 24 blades / 1 % clearance; calibrate hub/tip and angles on Murray's maps with held-out lines; uncertainty; low-speed efficiency; "calibrated, not measured" everywhere | D-061/062; `configs/components/kj66_turbine.yaml`. **Low-speed efficiency does not match** (below) |
| 3 | Re-run the idle-EGT check | Full KJ-66 replica (real compressor + calibrated turbine, A9 fitted at the top speed only) |
| 4a | Ingest Nickel 337 (table + LM figure), label, cross-check vs 393, anchors, AMS 5391 | Transcribed tables + digitized Fig 7 (D-063). All checks pass |
| 4b | TIT > 1150 K infeasibility: missing data or physics? | **Physics for ~90 %** (below) |
| 5 | v5 pilot → full if ≥ 10 % | Pilot 30.5 % → full 2000 run (1277 s, 3 workers) |

## 1. Compressor refit (R1) and extrapolation flag (R3)
Refit: Δσ −0.1041 → **−0.1176**, Cf 0.0937 → **0.1002**. Held-out lines (mean per line; `validation/phase4/compressor_refit_murray.png`):

| line | U2 [m/s] | PR err | flow err | η err |
|---|---|---|---|---|
| 40 krpm | 138 | −2.0 % | −5.9 % | +14.2 pts |
| 60 krpm | 207 | −1.8 % | −2.2 % | −0.6 pts |
| 80 krpm | 277 | −1.6 % | −0.7 % | −2.3 pts |
| 100 krpm | 346 | +0.6 % | +0.2 % | +0.6 pts |
| **120 krpm** | **415** | **+5.0 %** (was +7.1) | **+4.4 %** (was +6.1) | **+6.1 pts** (unchanged) |

The refit improves the top lines but cannot remove the trend: two constant factors give −1.5 … −1.9 % PR at 50–90 krpm
and +2.4 % / +5.0 % at 110 / 120 krpm. This is model form, not calibration. Calibration lines: 50 −1.5 %, 70 −1.9 %,
90 −1.2 %, 110 +2.4 % PR. Inflated band k = 3.5 (was 2.75) because the held-out set now reaches 40 and 120 krpm.

R3 justification: the signed PR error per line grows like a quadratic in U2 (fit residual 0.19 %). Extrapolated:
+7.0 % at 440 m/s, +8.9 % at 460, +10.9 % at 480, +13.1 % at 500. The band is widened by W = e(U2)/e(415): 1.43 /
1.81 / 2.22 / 2.67, only toward lower slip and higher diffuser loss (the side the measured bias points to).

## 2. KJ-66 turbine calibration (calibrated, not measured)
- **Adopted (fit F: Murray flow lines 26/38/50/62 krpm + Appendix D exit velocity V5 on odd D-points):**
  - hub/tip **0.743** (profile range 0.65–0.745);
  - NGV gauging angle **64.8°** (64.8–71.1°);
  - rotor gauging angle **43.6°** (36.1–43.6°);
  - rotor metal inlet angle 27.3° (zero incidence at 120 krpm, a rule);
  - consequences: blade height 8.5 mm, hub diameter 49 mm.
- **Uncertainty is large and the fit is barely identifiable.** The flow map alone fixes only the NGV throat area: a
  flow-only fit gives hub/tip 0.52 ± 0.60. Hub/tip comes from V5, assuming station 5 is the rotor exit plane. The
  Jacobian sigmas are meaningless (condition number 1e8), so the ranges above come from a profile scan. Above hub/tip
  0.745 the model cannot reach the measured points.
- **Held-out errors (F):**
  - flow +10.9 % (20 krpm), +3.0 (32), −0.7 (44), −7.0 (56), −10.4 % (68);
  - V5 −3.5 % mean, 5/7 within uncertainty.
- **Low-speed efficiency: NO, it does not match Murray.**
  - F is +11.2 pts above Murray's (derived) turbine efficiency over all lines, +10.9 pts at ≤ 44 krpm corrected; the
    flow-only fit Q is +12.9 pts.
  - Fitting efficiency as well (fit FE) pushes hub/tip to the 0.40 bound, with 48° exit swirl and V5 −38 %. That is
    geometry used as a loss knob, so it was rejected.
  - **The gap is in the loss model or in Murray's derived efficiency (±0.08–0.34), not in the geometry.**
- **Caution:** the counts are from a published KJ66 variant (Xu et al.), not certainly Murray's engine.

## 3. Idle EGT: KJ-66 replica
Built from the real compressor (R1 calibration) and the calibrated turbine, with A9 fitted per run at the top speed;
38 validation points (`validation/phase4/kj66_replica_idle_egt.png`).

| turbine | EGT err, all | EGT err, < 50 krpm | model TIT < 50 krpm | air flow | P03 |
|---|---|---|---|---|---|
| F (adopted) | +52 K | +78 K | 877 K | −14 % | −1.8 kPa |
| Q (flow-only) | **−13 K** (rms 18) | **−13 K** | **785 K** | −11 % | −2.1 kPa |
| round 4 replicas | | +40 … +80 K | 840–860 K | | |

- With the turbine that reproduces Murray's flow map (Q), the hot idle is gone: low-speed TIT is 785 K against
  Murray's 780–820 K.
- Lowering the turbine efficiency to Murray's level (diagnostic only) makes idle hotter, not cooler. So the earlier hot
  idle came from turbine flow capacity, not from the optimistic efficiency.
- Caveats:
  - Both replicas under-predict air flow (−3 % at the top, −13…−17 % at idle). The model speed lines are almost flat,
    so engine flow follows the turbine + nozzle capacity.
  - Q's geometry contradicts V5.
  - Q's agreement may partly be compensating errors.
- The adopted F geometry is still 60–90 K hot at low speed.

## 4. Nickel 337 and "missing data or physics"
- **Ingested:**
  - p. 7 rupture table (10 / 100 / 1000 / 10,000 h, 1350–2000 °F; parentheses = extrapolated, excluded);
  - Table II individual tests [MEAS];
  - AMS 5391 minima and melting range (all transcribed);
  - Fig 7 Larson–Miller curve (digitized).
- **Checks:**
  - identical to Nickel 393 at all 7 common points;
  - your anchors match (1350 °F: 76 ksi / 1000 h, 56 ksi / 10,000 h, (97) ksi / 100 h extrapolated);
  - AMS 5391 passes: the 1800 °F / 22 ksi test lasted 65 h, and the table, LM curve and model give 72 / 67 / 49 h,
    all ≥ 30 h.
  - The LM master curve over-predicts the 2000 °F tests by 12–37 %, so it is not used to extend the data.
- **TIT > 1150 K, infeasible designed engines: it is PHYSICS.**
  - v4: 141 of 157 (90 %); v5: 131 of 147 (89 %).
  - With both alloys, the rotor's centrifugal stress × 1.5 exceeds the 100 h rupture strength at the rotor-relative
    temperature (1086–1314 K, inside the data). In about 32 of these the IN-713C vane hot streak is also above the
    alloy's solidus (1533 K): it would melt.
  - **More rupture data will not open these designs. That is an acceptable result.**
- **The remaining 16 (both datasets) are MISSING DATA:**
  - the MAR-M 247 vane hot streak (1342–1588 K) is above its 1311 K data;
  - Nickel 337 covers only IN-713C up to 2000 °F, so it cannot resolve them;
  - 7 of the v5 ones would pass with an IN-713C NGV + MAR-M 247 rotor, a mixed material set the model does not use.
- Nickel 337 changes nothing at the 100 h life; the creep model keeps Nickel 393 (D-063).

## 5. Dataset card and feasibility statistics: `phase4_v5` (2000 designs, full run)
Card: `docs/dataset_cards/phase4_v5.md`.

| | count |
|---|---|
| designs | 2000 (train/val 1700, **test 300, frozen before evaluation**) |
| design failed | 969: no buildable hub/tip 964, other turbine sizing 5; compressor 0 |
| designed engines | 1031 (PR 1.62–3.26, 19–254 krpm, design thrust 14–725 N among the feasible) |
| **feasible** | **604 (30.2 % of designs, 58.6 % of designed)**; test split: 93 |
| infeasible designed engines | AN² 188; AN² + NGV above data 179; NGV above data 38; rotor creep (± NGV) 21; rotor + NGV above data 1 |
| material status | IN-713C 515, MAR-M 247 89, fails creep 388, no material data 39 |
| sensitivity (feasible) | At/Ah 0.34: 839; PF 0.15: 639; PF 0.35: 468 |
| off-design (designed) | design 1031/1031, M0.5/3 km 1027 (4 no net thrust), 80 % 1031, idle 1029 (2 compressor choke) |
| calibration-sampled | 399 designs (20 %) |
| cross-check vs pyCycle-CEA | 21 checked, **0 flagged** (max ΔFn 0.037 %, ΔTSFC 0.094 %) |

**Feasibility by tip-speed regime and TIT band:**
| regime / TIT band | designs | designed | feasible | % of designs | % of designed |
|---|---|---|---|---|---|
| validated (U2 ≤ 415 m/s), TIT ≤ 1150 K | 735 | 409 | 346 | 47.1 | 84.6 |
| validated, TIT > 1150 K | 131 | 70 | 2 | 1.5 | 2.9 |
| **validated, all** | 866 | 479 | 348 | **40.2** | 72.7 |
| extrapolated (U2 > 415 m/s), TIT ≤ 1150 K | 966 | 472 | 255 | 26.4 | 54.0 |
| extrapolated, TIT > 1150 K | 168 | 80 | 1 | 0.6 | 1.2 |
| **extrapolated, all** | 1134 | 552 | 256 | **22.6** | 46.4 |

- Extrapolated designs are less often feasible mainly because of AN²: higher tip speed means higher shaft speed for a
  given flow. That is physics, not the wider band; the design-success rate is 46.7 % with the sampled band vs 49.2 %
  without.
- v4 shows the same split (363 vs 262 feasible).
- 42 % of the feasible v5 engines are extrapolated: the surrogate must carry the flag.

## Open issues
- Compressor model form: the error trend with speed remains after R1 (+5 % PR at 120 krpm). Every design above
  415 m/s is labelled; the surrogate should use the label (or a separate error model).
- Turbine efficiency at KJ-66 scale: the model is 11–13 pts above Murray's derived values. It affects every dataset
  turbine (sized by the same loss model) as a likely optimism in η_t.
- Turbine flow capacity vs speed: no single geometry matches both the flow map and V5; the engine replica's air flow
  is 11–17 % low at idle.
- MAR-M 247 above 1900 °F (16 missing-data designs per dataset); mixed NGV/rotor materials not modelled.
- Cross-check of the counts (18/24, 66 mm) against Murray's own engine is still missing.

## Proposals (not implemented)
- **R2, inducer choke / shock loss.** Published loss models exist:
  - Aungier, *Centrifugal Compressors: A Strategy for Aerodynamic Design and Analysis* (ASME 2000), choking loss;
  - Galvas 1973, NASA TN D-7487 (both found cited in meanline-validation literature).
  The KJ-66 inducer relative tip Mach rises from 0.67 (90 krpm) to 0.92 (120 krpm), so designs above 415 m/s are
  transonic, which is where such a term acts.
  - Proposal: implement the Aungier choking loss from the primary source and refit on the same split.
  - **I have not read either primary source.** I'd need the book chapter or the NASA report to take the equations
    from (the NASA report is public on NTRS; I can fetch it if you agree).
- **T-η, turbine efficiency calibration.** One effective loss multiplier on the Benner profile + secondary losses,
  fitted on Murray's efficiency lines with held-out lines, like the compressor's Cf. Cheap, but Murray's efficiency is
  derived (±0.08–0.34).

## Decisions I need from you
1. **R2:** fetch Galvas (NASA TN D-7487) / use Aungier and propose the exact choking-loss term? Or keep R3 only?
2. **Turbine efficiency (T-η):** add a calibrated loss multiplier to the turbine (it changes every dataset turbine and
   would need a v6), or keep it as an open item?
3. **KJ-66 turbine geometry:** keep F (fits V5, idle 60–90 K hot) as the reference, or Q (fits the flow map and the
   idle, contradicts V5)? Any source for the KJ-66 turbine hub diameter or blade height would settle it.
4. **Go for Phase 5** (surrogates) on `phase4_v5`.

# Phase 4, round 6 (2026-10-27): R2 test (Galvas), turbine loss multiplier, T-A / T-B, v6 prepared

Phase 5 pipeline status: `docs/PHASE5_REPORT.md` (built and tested on v5; provisional).

## (a) R2: Galvas (1973), NASA TN D-7487 — fetched, tested. **Proposal: do not adopt**
- Fetched from NTRS without problems (no allowlist change needed).
- **The report contains no loss term that grows with inducer relative Mach or tip speed.**
  - p. 2: "Shock losses in the rotor are neglected."
  - p. 10: "For inducers with relative Mach numbers appreciably in excess of unity the predicted compressor total
    pressure ratio and efficiency are greater than that attainable experimentally. This is a result of neglecting
    shock losses in the inducer inlet."
- Mach-dependent pieces that do exist (equations and tables in D-065):
  1. An **inducer choke criterion**: choking incidence vs rms relative Mach for rms blade angles 40–60°
     (listing p. 26/29). This is a flow limit, not a loss.
  2. A **vaned-diffuser recovery table** vs throat Mach. Recovery *rises* with Mach and is subsonic only.
     It needs vane geometry that is unpublished for the KJ-66.
- **Test on Murray's held-out lines:**
  - Galvas never flags choke. At 120 krpm (inducer tip relative Mach 0.92, rms 0.72) the incidence is 15° away
    from the choking incidence.
  - **Held-out errors with the term = without it** (PR +5.0 % / flow +4.4 % / η +6.1 pts at 120 krpm).
  - On v5 it would flag 4 of 1031 designs.
- So Galvas gives nothing that explains or corrects the high-speed bias. Keep R3 (flag + widened band).
- Aungier (2000) was not accessible here. If you can provide its choke/shock-loss section, I can test it the same way.

## (b) Turbine loss multiplier k_loss (approved; fitted; held-out errors)
One factor on the total turbine loss coefficient, fitted on Murray's efficiency lines 26/38/50/62 krpm with
32/44/56/68 held out. Geometry is fixed at each variant.

| variant | k_loss | held-out η error: published losses → fitted | held-out flow error |
|---|---|---|---|
| T-A | **1.955** (1σ 1.8–2.0) | +11.3 → **+1.7 pts** (rms 3.5; every held-out line within Murray's uncertainty) | −5.7 → −9.4 % |
| T-B | **1.762** (2σ 1.6–1.7) | +12.8 → **+6.6 pts** (rms 7.2) | −1.7 → −5.9 % |

**v6 proposal:** k_loss = **1.86**, band **[1.6, 2.0]**. The band is the union of the two variants' intervals, so
it carries the geometry spread. The 20 % calibration-sampled designs draw k_loss in the band.

Caveats:
- The meanline solver fails to converge at isolated high-loss / high-speed points.
  - The fit uses a bracketing workaround.
  - The upper edges of both intervals are set by such failures (T-A: a bracket jump at 50 krpm; T-B: the 68 krpm
    line cannot reach the measured PR).
- More loss lowers the fixed KJ-66 geometry's flow capacity, because the geometry was fitted with k = 1. A joint
  geometry + k refit would fix this; the dataset is not affected, because its turbines are sized per design.

**Effect on the dataset (probe: 120 v5 designs re-evaluated, not a regeneration):**
- design turbine efficiency −7.3 pts mean;
- design TSFC +19 % mean (+6 … +100 %);
- 3 of 61 designed engines become design failures, 1 feasible → infeasible;
- 6 cruise points newly fail, 3 of them `turbine_no_convergence`.

## (c) T-A / T-B bracketing variants
| variant (k_loss) | idle EGT err | idle TIT | thrust @ 120 krpm | TSFC @ 120 krpm [g/kN/s] | η_t,ts @ 120 krpm |
|---|---|---|---|---|---|
| T-A (1) | +78 K | 877 K | 56.1 N | 43.5 | 0.66 |
| T-B (1) | **−13 K** | **785 K** | 53.9 N | 47.1 | 0.79 |
| T-A (1.955) | +109 K | 897 K | 47.4 N | 50.4 | 0.61 |
| T-B (1.762) | **−0.3 K** (1 run) | 815 K | – | – | – |

- The geometry choice matters most for idle (~90–110 K in EGT) and for turbine efficiency.
- Thrust and TSFC at the top speed differ by 4–8 %.
- With the fitted k_loss the replica cannot run 2 of 3 runs per variant at the top speed (`turbine_no_convergence` at
  every nozzle area). That is the same solver weakness again.
- Hobbyist hint (hub/tip ≈ 0.56, *hobbyist CAD, unverified*) is closer to T-B (0.52) than to T-A (0.74).
- Measured thrust is not in our data, so thrust is not validated.

## v6: prepared, NOT generated
`configs/dataset_phase4_v6.yaml`:
- v5 + k_loss 1.86 (`engine_overrides`) + per-design k_loss sampling in [1.6, 2.0];
- new seed 20261117, own frozen split;
- R2 not included, per the proposal.

Generator support is tested; v1–v5 samples are unchanged (bit-identical). Run command, after your answer:
`uv run python scripts/phase4/pilot_then_full.py --config configs/dataset_phase4_v6.yaml`.

## Decisions I need from you
1. **R2:** agree not to adopt Galvas (it has no high-Mach loss), keeping R3? Or send Aungier's section for a test?
2. **v6 regeneration** with k_loss 1.86, band [1.6, 2.0] (D-064): go?
3. **Turbine solver robustness before v6 (recommended).**
   - At high loss the meanline root finder fails at isolated points: ~5 % of probe designs gained a failure, and
     the replica's top speed fails.
   - Option: add a retry with a different starting pressure ratio / bracket inside the engine's turbine evaluation
     before declaring `turbine_no_convergence`. It is numerical only: results change only where the solver now
     fails. Do it before v6, or accept the failures (they are recorded as data)?
4. Optional: joint refit of the KJ-66 turbine geometry together with k_loss (fixes the flow-capacity side effect on
   the reference engine; does not affect the dataset).

# Phase 4, round 7 + end of Phase 5 (2026-10-27): v6 generated, surrogates retrained, joint turbine refit

## What was done, item by item
| # | Your instruction | Done |
|---|---|---|
| 1 | R2: record why Galvas was rejected (held-out evidence); Aungier open | D-065a; Aungier = open item |
| 2 | v6 with k_loss 1.86, band [1.6, 2.0]; sanity-check turbine efficiencies first (range from Murray and DePaola); k_loss sampled for the ~20 % subset; held-out turbine errors in DECISIONS | Range [0.58, 0.93] from Murray (**DePaola's turbine-efficiency values are not in our extracts**; please send if available), D-069. Pilot: 0 flagged; full v6: 0 flagged. Held-out turbine errors in D-064 |
| 3 | Turbine solver restarts (2–3), retry count + converged_on_retry logged, failures kept; report v5 pilot recoveries | D-067. **v5 pilot: nothing to recover** (no turbine-solver failures at k_loss 1); at k_loss 1.86, 1 of 9 recovered |
| 4 | Joint KJ-66 geometry + k_loss refit with held-out lines; closer to T-A or T-B; resolves the conflict? | D-068. Closer to T-A (best RSS). **Does not resolve the conflict** |
| 5 | v6 pilot → full; Phase 5 retrain on v6; compare with v5 | v6: 2000 designs, 568 feasible (28.4 %). Surrogates retrained (one command) |

## v6 dataset card and statistics (`docs/dataset_cards/phase4_v6.md`)
| | v5 | v6 |
|---|---|---|
| designed / feasible | 1031 / 604 (30.2 %) | 1028 / **568 (28.4 %)** |
| feasible, validated / extrapolated tip speed | 40.2 % / 22.6 % | 38.0 % / 21.0 % |
| feasible, TIT > 1150 K | 1.0 % | 1.3 % |
| design η_t,tt (median, range) | – | 0.789 (0.62–0.86), **0 outside [0.58, 0.93]** |
| design η_t,ts median | 0.71 (pilot) | 0.64 (pilot) |
| feasible design TSFC best / median [g/kN/s] | 43.9 / 61.3 | 43.2 / 72.5 (JetCat claims 40–52) |
| cruise point converged | 1027 / 1031 | 946 / 1028 (38 no net thrust, 37 turbine non-convergence) |
| converged on retry / failed after retries | – | 3 / 44 |
| k_loss subset | – | 397 sampled in [1.6, 2.0] (114 feasible); 1603 nominal 1.86 (454 feasible) |
| pyCycle cross-check | 0 / 21 flagged | 0 / 21 flagged |

## Surrogates on v6 vs provisional v5 (frozen test sets)
Details: `docs/PHASE5_REPORT.md`, `validation/phase5/compare_phase4_v5_vs_phase4_v6.md`.

| target (ensemble) | median relative error: v5 → v6 |
|---|---|
| Fn design | 0.33 → 0.60 % |
| Fn idle | 0.43 → 0.59 % |
| T05 idle | 0.21 → 0.30 % |
| **Wf design (new)** | 0.36 % |
| **Wf cruise (new)** | 0.58 % |

- 95 % coverage 0.93–1.00 for all targets.
- TSFC targets have heavy tails in v6 (low-thrust designs; design TSFC R² 0.63, cruise 0.06). **Derive TSFC as
  Wf/Fn.**
- Classifier: accuracy 0.953 → 0.923, AUC 0.976 → 0.970, ECE 0.033 → 0.032.

## Joint KJ-66 turbine refit (D-068)
| start | hub/tip | k_loss | held-out η / flow | V5 (not fitted) | idle EGT |
|---|---|---|---|---|---|
| near T-A | 0.735 | 1.95 | +2.2 pts / −1.2 % | −5 % | +100 K |
| middle (lowest RSS) | 0.649 | 2.33 | +1.7 pts / −1.1 % | −29 % | +87 K |
| near T-B | 0.482 | 2.10 | +3.8 pts / −3.0 % | −51 % | +29 K |

- The maps don't identify the geometry: all three fit about equally well.
- The best fit is closer to T-A.
- Fitting jointly removes the flow penalty of the fixed-geometry k_loss fit.
- **It does not resolve the exit-velocity vs idle-EGT conflict:** whatever matches V5 is hot at idle. The cause is
  probably outside the turbine geometry.
- The joint fits want k_loss 1.95–2.33, at or above the v6 band.

## Open issues
- Idle EGT vs exit velocity for the KJ-66 reference engine (see above).
- `turbine_no_convergence` at high loss: structural in the meanline's regime logic. It costs ~4 % of the cruise points
  in v6. A fix needs a change in the meanline (not done).
- DePaola's turbine-efficiency value (not in our extracts); Aungier (R2) not accessible.
- k_loss: joint fits suggest 1.95–2.33 vs the v6 band [1.6, 2.0].

## Decisions I need from you
1. **k_loss band:** keep [1.6, 2.0] (fixed-geometry fits), or widen to [1.6, 2.35] to cover the joint fits (would be
   a v7)? I'd keep v6 for Phase 6 and note the band as an uncertainty.
2. **Idle-EGT conflict:** pursue it (test low-power combustion efficiency, the station-5 assumption, the replica's
   low air flow), or accept it as a known limit of the KJ-66 reference case?
3. **Meanline regime fix** for the remaining turbine non-convergence (~4 % of cruise points): do it, or keep the
   failures as data?
4. **Surrogate set for Phase 6:**
   - targets: Fn, Wf (→ TSFC), T05, mass, diameter + P(feasible);
   - models: ensemble for thrust/temperatures, GP for fuel flow and geometry, chosen on validation data only.
   - Agree?
5. **Go for Phase 6** (multi-objective optimisation with physics verification of every Pareto design).

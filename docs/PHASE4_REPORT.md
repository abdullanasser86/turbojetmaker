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

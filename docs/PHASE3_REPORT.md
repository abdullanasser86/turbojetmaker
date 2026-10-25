# Phase 3 report: coupled engine model (2026-10-25)

> Preliminary design only, not manufacturing-ready. Everything below is 0-D/meanline physics.

## Your pre-Phase 3 decisions: what was done

| # | Decision | Status |
|---|---|---|
| 1 | KJ-66 calibration as provisional defaults, with source, fit points and band; separate calibration/validation sets | Done. `configs/components/kj66_compressor_calibration.yaml` stores the value, 1σ, 2σ band, fit points and split. The split is also logged in `docs/VALIDATION.md`. Phase 4 sampling spec: uniform inside the 2σ band for a fraction of designs (proposal 20 %, decision below). Every new compressor uses these factors. Recalibration waits for the new data. |
| 2 | Own fast meanline turbine; < 10 ms; ≥ 200 Sobol samples + Kofskey; TurboFlow kept; generic interface | Done, see results. 7.5 ms median in sweeps and engine iterations; a single cold point takes ~17 ms. |
| 3 | TIT bound 1350 K; material constraint from `configs/materials.yaml` | Done. IN-713C (default, 1253 K) and MAR-M 247 (premium, 1273 K), sourced extracts. "Premium material required" tag. Above all limits → infeasible but kept. |
| 4 | Ingest DePaola, Murray, Xiang thesis, JetCat pages | **Not possible: these files never arrived** (`/root/.claude/uploads` only holds the six PDFs from before). The DePaola values you summarised are used and labelled `user_summary`. |
| 5 | Repo, CUDA torch, env check, workers, checkpoints, README | Done. torch 2.10.0+cu128 + GPyTorch 1.15.2 + BoTorch 0.18.1 locked; `scripts/check_env.py` reports GPU/VRAM/driver/CUDA/power limit and runs a GP + MLP benchmark with peak memory. Workers default to physical cores − 1 (`compute.n_workers`). Resumable checkpoints are in `turboml/checkpoint.py`, and the README has a laptop-setup section. Pushed at the end of the phase (see below). |

## What was built

| Module | What it does |
|---|---|
| `turboml/physics/turbine_meanline.py` | Fast single-stage axial turbine. TurboFlow's equations (Benner losses ported verbatim, Aungier deviation, critical-Mach choking) with a marching solver and our real-gas thermo (combustion products). Handles all four choke regimes; failures come back as labelled data. Generic `TurbinePoint` and `TurbineModel` interface, so a radial turbine can be added later. |
| `turboml/physics/turbine_design.py` | Stage sizing from φ, ψ, R (velocity triangles → angles, heights, Zweifel pitch, throat). `design_stage` makes the stage pass exactly the design flow at exactly the design work. |
| `turboml/physics/engine.py` | Coupled engine. **Design mode:** compressor sized for the target PR (a KJ-66-like wheel), turbine sized for the compressor power, nozzle area from the design flow. **Off-design:** speed-controlled matching (unknowns air flow, TIT, turbine PR; Newton/Broyden with continuation). Labelled failures. Design card with the safety note. |
| `turboml/physics/mechanical.py` | Material temperature check with premium tag, blade root stress (AN²) proxy, impeller tip speed, envelope (D, L) and dry-mass proxies. |
| `turboml/physics/turbine.py` | TurboFlow wrapper, now with `AxialTurbine.from_geometry(...)`, kept as the higher-fidelity option. |
| configs | `materials.yaml`, `mechanical.yaml`, `engine_baseline.yaml`, `components/axial_turbine_design.yaml`; `bounds.yaml` updated (TIT 1350 K, turbine φ/ψ/R bounds) |
| scripts/phase3 | `turbine_port_verification.py` (Sobol), `kofskey_fast_turbine.py`, `baseline_engine.py`, `compressor_calibration_band.py`, `fit_envelope.py`, `make_notebook.py` |
| `notebooks/03_coupled_engine.ipynb` | Teaching notebook with all Phase 3 results (executed). |

## Validation and verification results

### Fast turbine
**Sobol verification vs TurboFlow (acceptance ±2 pts / ±2 % / ±2 %): accepted.** 256 samples across the design
space (φ, ψ, R, aspect ratio, clearance, work, mass flow, speed, T0 1000–1350 K, p0), each at the design estimate and
one off-design point: 454 points compared.
| | mean abs | 95th pct | max abs |
|---|---|---|---|
| η_ts | 0.05 pts | 0.11 pts | 0.47 pts |
| mass flow | 0.05 % | 0.11 % | 0.15 % |
| work | 0.04 % | 0.11 % | 0.65 % |
**Worst cases** (`validation/phase3/turbine_port_summary.yaml`): all off-design at 63–82 % speed with a choked
stator and +8…+23° rotor incidence, mostly φ 0.6–0.9 and R 0.2–0.4. Not counted: 27 samples gave impossible geometry
(hub radius ≤ 0) and 4 points failed in TurboFlow only. All are recorded in `turbine_port_sobol.csv`.
Speed: TurboFlow 18.8 s per point; fast model 7.5 ms in sweeps, ~17 ms cold (34 ms while sharing the CPU with
TurboFlow). With `rotor_plus`, a few points jump to a very different solution (η_ts up to 47 pts away), so that
option needs more work before it could become a default.

**Kofskey (TurboFlow geometry, 305 map points per mode, all converged).** Max deviation from TurboFlow is 0.10 % in
mass flow, 0.06 pts in η_ts and 0.12 % in power. Against the measured data (untuned): η_ts −1.20 pts mean (2.84 max),
mass flow +1.08 % (2.59 max), torque −0.53 % (5.06 max). That is the same model error as TurboFlow in Phase 2, as it
should be.

**Finding: TurboFlow's rotor incidence sign.** TurboFlow 0.1.18 intends a +1 incidence sign for rotors, but a bug
(`cascade_type = ["cascade_type"]`) applies −1 to every row. Using the intended sign makes the Kofskey fit *worse*:
η_ts −2.0 pts mean, −2.6 pts at 50 % speed where the rotor sees +27° incidence. So the default stays TurboFlow's
effective behaviour (`incidence_sign="turboflow"`), and the intended sign is available as `rotor_plus`. Which one
Benner et al. (1997) define is open (decision 1).

### Compressor calibration band
Fitted on the 10 calibration rows (90 krpm): Δσ = −0.1041 ± 0.0074, Cf = 0.0937 ± 0.0039 (1σ). The 2σ band is narrow
because it is statistical only. Held-out errors (PR max 4.6 %, η 3.1 pts) show the model-form error is larger.

### Coupled engine: baseline ~150 N (`validation/phase3/baseline_design_card.md`)
Design inputs: 0.30 kg/s, PR 3.0, TIT 1150 K, 130 krpm, turbine φ = 0.7, ψ = 1.0, R = 0.5. Sizing takes ~11 s;
an off-design point takes 0.6–7 s.

| | Fn [N] | TSFC [g/(kN s)] | m [kg/s] | PR_c | TIT [K] | EGT [K] |
|---|---|---|---|---|---|---|
| design (SLS) | 148.3 | 41.5 | 0.300 | 3.00 | 1150 | 1033 |
| M 0.5, 3 km | 99.5 | 48.4 | 0.257 | 3.05 | 1087 | 968 |
| 80 % speed | 72.3 | 52.0 | 0.216 | 2.05 | 1007 | 931 |
| 60 % speed | 33.3 | 73.8 | 0.146 | 1.50 | 954 | 912 |
| idle 35 % | 9.5 | 139 | 0.077 | 1.15 | 942 | 927 |

- **Verification:** at design speed the off-design solver reproduces the design point (1e-6). The 0-D cycle with the
  same component efficiencies gives identical Fn and TSFC; pyCycle-CEA is within 0.02 % / 0.075 %. The operating line
  converges at 14 of 15 speeds; 30 % fails with `no_nozzle_pressure_ratio` (labelled).
- **Sanity vs database** (32 engines, 104–193 N): specific thrust 494 (range 352–511), TSFC 41.5 (38–54.5, median
  46.4), 130 krpm (median 126.5k), Ø115 mm (90–120), L 298 mm (198–346), mass 1.7 ± 0.2 kg (proxy). EGT 759 °C, at the
  top of the 680–760 °C range and slightly above JetCat's 750 °C maximum.
- **Mechanical:** IN-713C feasible (103 K margin), impeller tip speed 482 m/s (limit 500). The blade root stress proxy
  is 557 MPa untapered, reported as **unchecked**: no sourced allowable stress yet.
- Part-load behaviour became physical only after making the off-design pressure-loss fractions scale with corrected
  flow² (D-032). With constant fractions, TIT rose to 1602 K at 50 % speed.

### Tests
87 tests pass (`uv run pytest`, ~4 min; 69 fast with `-m "not slow"`). New: turbine port vs TurboFlow map points and
loss function, warm-start invariance, products gas, failure labels, sizing, mechanical proxies, calibration split,
engine design/off-design/labels/design card.

## Open issues
1. **Incidence sign (D-029)**: needs Benner, Sjolander & Moustapha (1997), ASME J. Turbomach. 119:193–200, to check
   the angle convention. The TurboFlow bug should be reported upstream (together with the Phase 2 compressor bugs).
2. **Materials data missing**: stress-rupture / allowable stress for IN-713C and MAR-M 247, so AN² is unchecked. With
   0 K gas-to-metal allowance and the sourced use limits, **every design above 1273 K TIT is infeasible**. AMT-class
   engines stay in the sampled space, but they can only become feasible with a sourced allowance or a sourced
   higher-temperature material.
3. **Turbine efficiency at micro scale is unvalidated**. The baseline gives η_tt 0.895 / η_ts 0.742, but Kofskey is a
   0.24 m turbine. Measured micro-turbine or engine data (DePaola) is needed.
4. **Idle too hot**: idle EGT 654 °C vs JetCat's ~480 °C minimum. Design EGT is slightly above JetCat's maximum.
   Suspects: part-speed compressor efficiency (KJ-66 calibration only covers 70–110 krpm on the KJ-66), the loss-scaling
   rule, and turbine incidence at low speed.
5. **Speed**: the turbine meets < 10 ms in sweeps but not as a single cold point (~17 ms). The **compressor** is now the
   bottleneck (0.1–0.27 s per point, mostly the 1-D diffuser integration). A Phase 4 dataset at ~11 s design plus ~3 s
   per off-design point is ~20 s per design → ~2 h per 1 000 designs on 3 workers. A compressor speed-up is planned.
6. Small gap (< 0.3 % mass flow) between the turbine's subsonic and choked branches; handled consistently but noted.
7. Turbine blade counts come out non-integer (sizing); not rounded yet.
8. Surge flag (Senoo stall or positive slope) is still unvalidated; no surge data.

## Decisions I need from you
1. **Incidence sign:** keep TurboFlow's effective sign as default (my recommendation, it fits Kofskey data) until the
   Benner paper is checked? Can you supply the paper? And may I open issues on the TurboFlow GitHub (outward-facing)?
2. **Materials:** please supply sourced stress-rupture data (e.g. Nickel Institute "High-Temperature, High-Strength
   Nickel Base Alloys", or ASM Handbook) and decide whether to add a gas-to-metal allowance (needs a source) or a
   higher-grade alloy for > 1273 K.
3. **Phase 4 calibration sampling:** what fraction of designs should sample Δσ and Cf inside the band? I propose 20 %.
   Should the band be widened to cover model-form error? (That needs data, or must be labelled an assumption.)
4. **DePaola values:** confirm that "nozzle ≈ 0.95" means the nozzle total-pressure ratio. Confirm the off-design
   loss-scaling rule (losses ∝ corrected flow²).
5. **Your OS** on the RTX 4080 laptop. On Windows the CUDA torch build needs the PyTorch package index in the lock,
   and I need to know before Phase 4 training.
6. **Documents:** please re-upload DePaola 2020, Murray 2009, Xiang 2015 (Appendix A-I) and the JetCat pages; they
   didn't arrive. jetcat.de, apps.dtic.mil, dr.ntu.edu.sg and scholar.afit.edu are blocked from here.
7. **Phase 4 scope:** which off-design points to store per design (proposal: design + M0.5/3 km + 80 % + idle), and
   the first dataset size (proposal: 2 000 designs).
8. "Go" for Phase 4.

# Phase 1 report: cycle model (2026-10-24)

## What was built
| File | Purpose |
|---|---|
| `turboml/gas/thermo.py` | Ideal-gas mixtures (air, complete-combustion products of C12H23) with variable cp from NASA 9-coefficient data; fuel enthalpy from LHV + combustion efficiency |
| `turboml/physics/atmosphere.py` | ISA / US-1976 to 20 km, geometric altitude |
| `turboml/physics/cycle_common.py` | Shared `CycleInputs` / `CycleResult` (stations 0-2-3-4-5-9, failure labels) |
| `turboml/physics/cycle_simple.py` | **Independent 0-D design-point model**, ~5 ms per run, plus an air-standard perfect-gas mode for analytic tests |
| `turboml/physics/cycle_pycycle.py` | **pyCycle micro-turbojet** (design point): convergent nozzle, mechanical efficiency, shaft off-take, fuel enthalpy from LHV, SI in/out, CEA or TABULAR thermo, 0-D result as Newton initial guess, failures returned not raised |
| `configs/cycle_baseline.yaml` | Baseline inputs, each tagged assumption / datasheet / spec |
| `validation/cases/jetcat_datasheets.yaml` | 5 JetCat engines (unverified extracts, with URLs) |
| `scripts/phase1/cross_check_sweep.py`, `jetcat_inverse.py` | Acceptance sweep; datasheet forward + inverse analysis |
| `configs/bounds_proposal_phase1.yaml` | Proposed cycle-variable bounds (inactive until you confirm) |
| `notebooks/01_cycle_model.ipynb` | Executed notebook with plots |
| Tests | 29 new (40 total incl. slow solver tests); all pass |

## Acceptance criteria
| Criterion | Result |
|---|---|
| The two models agree within ~1 % on thrust and TSFC across a parameter sweep (matched assumptions) | **63/64 points pass with pyCycle-CEA; strictly, this is a FAIL.** Median differences are 0.03 % (Fn) and 0.06 % (TSFC); compressor power is identical. The one failing point (Fn 1.03 %, TSFC 0.93 %) is ill-conditioned: an engine making 36 N per kg/s at Mach 0.5, where net thrust is a small difference of large terms. There the difference is 0.18 % of gross thrust. With pyCycle-TABULAR: median 0.3 %, max 3.5 % → **fails**, caused by pyCycle's table interpolation (it differs from both CEA and 0-D by up to 1 % in compressor work alone). |
| Published engines within a plausible band (sanity check) | **Partly.** Thrust + SFC of P62/P160 (and P220) can be reproduced with plausible component efficiencies and an EGT inside the listed range. The P300/P400 extracted data is internally inconsistent with their listed EGT (see below). Forward prediction with the placeholder efficiencies is off by −19…+23 % in thrust and −10…−15 % in SFC. |
| Unit tests cover energy / mass balance | ✅ Whole-engine energy balance to 1e-10, shaft and mass balance, ideal-Brayton analytic limit to 1e-9, JANAF property values, failure labels |

Speed (this 4-core container): 0-D ≈5 ms per point; pyCycle TABULAR ≈1.1 s; pyCycle CEA ≈3.7 s (sweep averages).

## Findings
1. **pyCycle's default fuel enthalpy is 0**, which corresponds to an LHV of 44.85 MJ/kg. Used as-is, TSFC comes out
   ≈4.6 % too low. Fixed in our wrapper (D-010). This affects anyone using pyCycle's examples unmodified.
2. **pyCycle TABULAR thermo** is ~3× faster than CEA but costs up to 3.5 % accuracy in the micro-engine regime.
3. **pyCycle quirks** (all handled):
   - the atmosphere table is indexed by geopotential altitude (D-011);
   - the nozzle `choked` flag is never computed (D-012);
   - the carbon atomic weight is 12.017 instead of 12.0107 (negligible).
4. **Exhaust temperature is set by energy conservation**: FAR = thrust × SFC / mass flow, almost independent
   of component efficiencies. Applying this to the extracted P300-PRO / P400-PRO numbers gives an implied EGT of
   930–980 °C against a listed maximum of 750 °C. That would need combustion efficiency ≤ 0.70–0.74. The likely
   causes are an extraction error, a different definition of the listed mass flow or thrust, or EGT being a
   thermocouple reading rather than the mean total temperature. I can't tell which without the official datasheets.
5. **Placeholder efficiencies are optimistic.** Every engine burns 10–15 % more fuel per newton than the baseline
   predicts.

## Honest limits
- **Component efficiencies are still assumptions**, not sourced values. I could not reach journals, DTIC or
  jetcat.de from this container (only PyPI/GitHub are allowed), and I haven't read the KJ-66 paper.
- The JetCat data are second-hand web-search extracts.
- The pyCycle model is design-point only; its compressor and turbine maps are generic placeholders (D-013).
  Off-design comes in Phase 3 with the meanline maps.

## Decisions I need from you
1. **Cross-check acceptance.** Do you accept 63/64 points within 1 % (CEA), with the one exceedance explained by
   ill-conditioning?
   - (a) **Accept** *(recommended)*: the physics agreement is 0.03 % median, and the outlier is outside any
     realistic design.
   - (b) Change the metric to |ΔFn|/Fg, which passes everywhere (max 0.18 %). Changing a criterion after seeing
     the result needs your explicit OK.
   - (c) I dig further into the residual 0.1 % (equilibrium chemistry vs complete combustion) before moving on.
2. **Which cycle model generates training data (Phases 3–6)?**
   - (a) **The 0-D model as the workhorse, with pyCycle-CEA as a periodic cross-check** *(recommended)*. It is
     ~700× faster, agrees with CEA to ~0.1 %, and is easier to couple with our own meanline components and
     off-design matching in Phase 3.
   - (b) pyCycle-CEA everywhere: closest to the brief's wording, but about 4 s per point. Coupling meanline maps
     into pyCycle's map format is also more work.
   - (c) pyCycle-TABULAR: fails the 1 % criterion.
3. **Bounds** in `configs/bounds_proposal_phase1.yaml`: PR 2.0–4.5, mass flow 0.10–1.8 kg/s, TIT 1000–1250 K
   (provisional), shaft speed 60–200 krpm. Confirm or adjust.
4. **Data, which matters most for Phase 2:**
   - the micro-turbojet review PDF;
   - the official JetCat datasheet PDFs;
   - the KJ-66 compressor paper (Xiang et al. 2017);
   - any source of measured component efficiencies.

   Alternatively, allow journals.shahroodut.ac.ir, apps.dtic.mil and jetcat.de in the environment's network
   settings and I'll fetch them myself.
5. **Still open from Phase 0:** a GitHub repo to push to (everything is only in this container) and your
   workstation's GPU and driver.

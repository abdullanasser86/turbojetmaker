# Phase 2 report: component meanline models (2026-10-24)

This report covers the pre-Phase 2 items you asked for (1–5) and Phase 2 itself.

## Pre-Phase 2 items
| # | Item | Status |
|---|---|---|
| 1 | Accept 63/64; known exception | ✅ Sample 8: nozzle pressure ratio 1.09 and Fg/Fn = 5.6 give d ln Fn / d ln Pt9 ≈ 31, which amplifies a 0.03 % CEA-chemistry difference to 1 %. Documented in `docs/VALIDATION.md`, pinned in `validation/phase1/known_exceptions.yaml` with a regression test. Primary metric unchanged; |ΔFn|/Fg reported as a labelled **secondary** metric. |
| 2 | 0-D for data; cross-check rules | ✅ `turboml/physics/crosscheck.py`: random 2 % + all Pareto designs, parallel pyCycle-CEA, flag > 1 %, gate at 5 % (stop and report). TABULAR removed. 0-D model split into pluggable component functions. |
| 3 | Bounds | ✅ `configs/bounds.yaml` (PR 2–5, W 0.08–1.9 kg/s, N 50–220 krpm, TIT 1000–1250 K provisional, tip speed ≤ 500 m/s), each bound with its source; re-derived from the database (below). |
| 4 | Data ingestion | ✅ partly: 6 PDFs ingested (summary below). DePaola, Xiang's thesis and the P20–P400 product pages weren't in the upload. |
| 5 | GitHub + GPU | ❌ **Your message still had the placeholders** `<YOUR_REPO_URL>` and `<PASTE OUTPUT OF: nvidia-smi …>`. No remote is set; all commits are local to this container. `scripts/check_env.py` now verifies every key package against `uv.lock` and smoke-tests torch, GPyTorch and BoTorch on the auto-selected device. |

### What was extracted (item 4) → `validation/cases/` (see its README)
| Source | Extracted |
|---|---|
| Micro-turbojet review (Anjomrouz et al. 2023), Table 3 | **109 engines**, 15.6–979 N: thrust, D, L, rpm, PR (28), mass flow (18), fuel flow, EGT, per-row manufacturer reference. Parsed by script; spot-checked against the page images. |
| JetCat PRO catalog 2024 | 14 variants (P250 … P1000): rpm, thrust, EGT, PR, mass flow, exhaust velocity and power, fuel (ml/min and kg/min), SFC, mass, dimensions |
| JetCat P500/P550 datasheets (2023) | 2 rows |
| Xiang, Schlüter & Duan 2016 (the journal paper you uploaded) | KJ-66 geometry (Table 1, radii), CFD speed lines, and **measured** KJ-66 PR/η at 70–110 krpm (Murray 2009), **digitized** with stated uncertainty |
| Ling, Wong & Armfield 2007 | KJ66 impeller geometry (66 mm, inducer 12.5/45 mm, b2 6 mm, clearance 0.3 mm, blade-angle curves) and CFD peaks |

Source inconsistencies found: the review's Table 1 says PR 1.4–5.7 but its Table 3 tops out at 4.7; P400 is cited to
KingTech's website; P160 fuel flow doesn't match its SFC. Also, the "η 0.73 → 0.55" in the brief is Xiang's **CFD**. The
**measured** KJ-66 efficiency stays at ~0.73–0.77.

### Bounds re-derived from the database (item 3, last bullet): proposed updates
| Bound | Active | Database (≥ 50 N) | Proposal |
|---|---|---|---|
| PR | 2.0–5.0 | 2.0–4.7 (n = 27) | keep |
| Air flow | 0.08–1.9 kg/s | specific thrust 352–634 N·s/kg; < 250 N engines 352–511 → 50 N needs 0.10–0.14 kg/s | keep (covers it) |
| Shaft speed | 50–220 krpm | 61.5–205 krpm; fit 190 krpm @ 50 N, 68 krpm @ 1000 N | keep |
| TIT | 1000–1250 K | max EGT 650–875 °C; AMT engines list 875 °C (1148 K) | **raise the upper limit to ~1350 K, or document that AMT-class engines are excluded.** TIT > EGT, so 1250 K can't reach them. Waiting on DePaola's measured TIT. |

## Phase 2: what was built
| File | Purpose |
|---|---|
| `turboml/physics/compressor.py` | **TurboML centrifugal compressor meanline**: Oh (1997) impeller loss set (ported from TurboFlow, MIT), 1-D friction vaneless diffuser (real p0 loss), Wiesner/Stodola slip + Δσ, choke (inducer / throat / exit / diffuser) and surge criteria, speed lines. ~0.1 s/point, no CoolProp. |
| `turboml/physics/compressor_radcomp.py` | RadComp cross-check (independent loss set) |
| `turboml/physics/compressor_turboflow.py` | TurboFlow cross-check (with our slip-bug fix) |
| `turboml/physics/turbine.py` | TurboFlow axial-turbine wrapper (failures as rows, maps) |
| `turboml/gas/thermo.py` | ~10× faster (mixture coefficients pre-summed; identical results) |
| `configs/components/kj66_compressor*.yaml` | KJ-66 geometry with per-field provenance (measured / digitized / assumption) + calibrated values |
| `scripts/phase2/*` | KJ-66 validation study, Kofskey turbine validation, gas-error study, notebook builder |
| `notebooks/02_component_models.ipynb` | executed, 5 figures |
| Tests | 55 total (+15 since Phase 1), all pass |

## Acceptance criteria
| Criterion | Result |
|---|---|
| Compressor reproduces KJ-66 trends (PR and η vs speed) within a documented error | **Yes, after calibration.** Blind: PR +21 %, η +13 pts, because inputs are unknown (RadComp blind: +12 % PR). Calibrated on 90 krpm only (2 effective parameters), the **held-out** 70/80/100/110 krpm give **PR +1.1 % mean, ±4.6 % max; η +0.9 pts mean, ±3.1 max**. Trend of steep PR rise and almost flat peak η is reproduced. |
| Turbine reproduces at least one published case | **Yes, untuned.** Kofskey 1972 (NASA TN D-6967), measured: η_ts 1.4 pts, ṁ 1.1 %, torque 1.5 % mean abs. |
| Slip-factor uncertainty propagated and shown | ✅ Δσ ∈ {±0.05, ±0.10} bands on speed lines and at measured points. Δσ = 0.05 moves PR by ≈ 5 %. |
| Speed lines / map, surge and choke detection | ✅ Maps generated with labelled failures; choke detected. Surge uses a documented criterion that is **not validated** (no surge data). |

## Findings
1. **TurboFlow 0.1.18 centrifugal-compressor bugs:**
   - slip uses degrees as radians (NaN for |β2b| > ~33°);
   - the algebraic vaneless diffuser is lossless with the Oh set;
   - the 1-D diffuser crashes.

   Worth reporting upstream. Its axial-turbine module is fine.
2. **Meanline surge prediction is weak.** With TurboFlow's incidence reading, speed lines have no PR peak at all. Our
   blade-relative incidence gives peaks at low speed only; diffuser stall governs at high speed.
3. **Two independent loss sets over-predict the KJ-66 in the same direction.** That points to unknown inputs (backsweep,
   diffuser) rather than to either loss model.

## Open issues
- **Calibration transfer.** The calibrated Δσ = −0.10 and Cf = 0.094 are effective, single-engine values from digitized
  data.
- **Turbine speed.** TurboFlow takes ~3 s/point. Phase 3 matching for thousands of designs (each needing maps) would take
  days.
- **Turbine geometry for micro engines.** The validated case is a 0.24 m NASA rig turbine. No micro-turbine geometry or
  data is available yet.
- **Diffuser model form.** The KJ-66 diffuser is vaned; we model it as vaneless.

## Decisions I need from you
1. **Compressor defaults for arbitrary designs (Phases 3–6):**
   - (a) **KJ-66-calibrated values as provisional defaults, plus uncertainty** (Δσ ±0.05 around −0.10, Cf range),
     re-calibrated when more data arrives *(recommended)*;
   - (b) nominal uncalibrated values (Δσ = 0, Cf = 0.005). Known to over-predict the only measured engine by ~20 %;
   - (c) wait for DePaola's measured compressor data (4 engines) before Phase 3.
2. **Turbine model for Phase 3:**
   - (a) **write a fast TurboML axial-turbine meanline** (Kacker–Okapuu or Ainley–Mathieson/Dunham–Came losses), verified
     against TurboFlow and Kofskey like the compressor *(recommended: needed for speed)*;
   - (b) tabulate TurboFlow maps per design (≈3–5 min per design → too slow for 5k designs);
   - (c) generic scaled turbine map with a correlation-based efficiency (fast, least physical).
3. **TIT upper bound:** raise to ~1350 K, or keep 1250 K and exclude AMT-class engines.
4. **Data:**
   - the DePaola AFIT thesis (top priority: measured η_c, η_t, TIT for 4 engines);
   - Murray 2009 thesis (paired KJ-66 data, maybe a map with surge);
   - Xiang's NTU thesis (KJ-66 blade exit angle and diffuser geometry would remove the two calibration knobs);
   - the P20/P80/P100/P220/P400 product pages.
5. **The GitHub remote URL and the `nvidia-smi` output** (both arrived as placeholders).

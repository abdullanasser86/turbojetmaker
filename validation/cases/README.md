# Validation and reference data

Every file carries its source citation (per row or in a header). Uploaded PDFs are **not**
committed (copyright); parsers in `scripts/data/` regenerate the CSVs from them.

| File | Content | Source | Kind |
|---|---|---|---|
| `review_engine_table.csv` | 109 micro-turbojets 15.6–979 N: thrust, diameter, length, max rpm, PR (28 engines), mass flow (18), fuel flow, max EGT; per-row manufacturer source | Anjomrouz, Ghadiri & Imani (2023), *J. Solid Fluid Mech.* 13(5):59–76, Table 3; parsed by `scripts/data/parse_review_table.py` | manufacturer data via review |
| `jetcat_pro_catalog_2024.csv` | 14 JetCat PRO variants (P250…P1000): rpm, thrust, EGT range, PR, mass flow, exhaust velocity and power, fuel flow (ml/min and kg/min), SFC, mass, dimensions | JetCat PRO Engines catalog 2024, pp. 14–15 (STP ±3 %); parsed by `scripts/data/parse_jetcat_catalog.py` | manufacturer data |
| `jetcat_datasheets_p500_p550_2023.csv` | P500-PRO and P550-PRO datasheets (V1.1, 02/2023) | JetCat datasheets | manufacturer data |
| `jetcat_datasheets.yaml` | Phase 1 web extracts (P62, P160, P220, P300, P400), now cross-checked against the two sources above (see header) | web-search extracts | superseded where noted |
| `kj66_compressor_exp_murray2009.csv` | **Measured** KJ-66 compressor PR and adiabatic efficiency, 70–110 krpm | Murray (2009) MSc thesis, Queen's Univ., as plotted in Xiang et al. (2016) Figs. 6–7; **digitized** | experiment (digitized) |
| `kj66_compressor_cfd_xiang2016.csv` | KJ-66 RANS CFD speed lines 80/100/117 krpm | Xiang, Schlüter & Duan (2016) *Proc IMechE G*, doi:10.1177/0954410016644632; peaks from text, rest digitized | CFD |
| `kj66_compressor_cfd_ling2007_peaks.csv` | KJ66 CFD peak PR/efficiency at 80–120 krpm | Ling, Wong & Armfield (2007), 16th AFMC; digitized | CFD |
| `kj66_engine_specs.csv` | KJ-66 engine specs from Xiang (original design) and Murray Table 1.2 (ArtesJet version): a spread of builds | [X17] Xiang et al. 2017 Table 1 / thesis Table A-i.1; Murray 2009 Table 1.2 (values given by the user) | MFR |
| `microturbine_reference_assumptions.csv` | Component assumptions (DePaola, Murray), turbine design ranges (DePaola Table 2), AN² taper equation, 10 % same-model spread | DePaola 2020; Murray 2009 (values given by the user) | ASSUM / CFD / MEAS per row |
| `jetcat_specs_user_2026.csv` | JetCat P20…P550 max rpm, thrust, SFC, EGT band, idle rpm/thrust; stock P90 | jetcat.de pages and datasheets (values given by the user) | MFR |
| `combustor_pattern_factor.csv` | Combustor exit pattern factor | Fear 1972 NASA TM X-2476 (MEAS); US Patent 5,265,425 (ASSUM) | MEAS / ASSUM |
| `kj66_compressor_geometry.yaml` | KJ-66 impeller and diffuser geometry (per-field sources; unknowns `null`) | [X16], [L07] | geometry |

## Known inconsistencies in the sources
- The review's Table 1 summary gives PR 1.4–5.7, but its Table 3 contains PR 1.5–4.7 (28 engines report PR).
  The abstract says "< 5".
- The review lists P400 PRO-LN with reference [63] (KingTech's website); it's a JetCat engine.
- P160-RXi-B fuel flow in the review (468 g/min) doesn't match its SFC × thrust (421 g/min).
- KJ-66: the brief's "peak efficiency 0.73 → 0.55 from 80 to 117 krpm" is Xiang's **CFD**. The **measured** efficiencies
  stay at ~0.73–0.77 from 70–110 krpm, and the CFD under-predicts the measured PR at higher speeds (Xiang's own conclusion).

## Data types and weighting (user decision 2026-10-25)
Every value carries a data type: **MEAS** measured, **CFD**, **MFR** manufacturer claim / design spec, **ASSUM**
modelling assumption recommended by the source. Weighting when sources disagree: MEAS > CFD > MFR > ASSUM. Existing
files: `kj66_compressor_exp_murray2009.csv` MEAS (digitized), `kj66_compressor_cfd_*` CFD, `jetcat_*` and
`review_engine_table.csv` MFR, `kofskey1972_turbine_measured.csv` MEAS, `ni393_cast_rupture.csv` MEAS (typical test data).
Validation error floor against MFR data: same-make, same-model engines differ by up to 10 % (DePaola 2020, MEAS).

## Requested but not yet available
- DePaola (AFIT 2020): the user extracted the assumptions and ranges above; the measured per-engine polytropic
  efficiencies and TIT (Ch. 4, figures only) are NOT available.
- Xiang NTU PhD thesis (KJ-66 compressor design details, Appendix A-I).
- JetCat P20/P80/P100/P220/P400 product-page specs (P20-SX, P80-SE, P100-RX, P220-RXi are in the review table).
- Murray (2009): Table 1.2 specs and modelling assumptions are in; the measured engine curves (PR, mass flow, TIT at
  48-120 krpm, Figs 5.1-5.3, App. D) are in figures only and NOT available.

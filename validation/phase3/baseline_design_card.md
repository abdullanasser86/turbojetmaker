# TurboML baseline engine (Phase 3)

> **PRELIMINARY DESIGN ONLY - NOT MANUFACTURING-READY. Real micro turbines spin at 100 000+ rpm and need detailed 3-D aerodynamic, structural, rotordynamic, thermal and combustion analysis plus physical testing by qualified people before anything is built or run.**

Status: physics-evaluated design point (0-D cycle + meanline components); not surrogate output.

**Flag: turbine efficiency at micro scale unvalidated (Kofskey validation turbine is 0.24 m).**

## Design point

| quantity | value |
|---|---|
| net thrust | 148.3 N |
| air mass flow | 0.300 kg/s |
| specific thrust | 494 N s/kg |
| TSFC | 41.5 g/(kN s) (0.150 kg/(N h)) |
| fuel flow | 6.16 g/s |
| shaft speed | 130000 rpm |
| compressor PR / eta_tt | 3.000 / 0.751 |
| TIT | 1150 K |
| turbine PR_ts / eta_ts | 1.855 / 0.742 |
| EGT (turbine exit total) | 1033 K (759 C) |
| nozzle area A9 | 16.70 cm2 (subsonic) |

## Geometry (meanline)

| part | value |
|---|---|
| impeller tip radius r2 | 35.39 mm (tip speed 482 m/s) |
| inducer hub / tip radius | 6.56 / 23.63 mm |
| impeller exit width b2 | 4.65 mm |
| blades / backsweep | 10 / -21 deg |
| diffuser outlet radius | 51.5 mm (vaneless model) |
| turbine mean radius | 27.55 mm |
| turbine rotor tip radius / blade height | 34.44 / 13.79 mm |
| turbine blades (stator / rotor, non-integer sizing) | 28.1 / 22.5 |
| envelope D x L (proxy) | 115 x 298 mm |
| dry mass (proxy) | 1.71 +- 0.17 kg |

## Constraints and proxies

Creep (NiDI No. 393 rupture data; life and safety factor from configs/materials.yaml):

| material | rotor T_metal (T0,rel) | rotor stress centrifugal + gas bending | allowable | rotor | NGV gas bending | NGV PF_max | NGV |
|---|---|---|---|---|---|---|---|
| in713c | 1088 K | 557 + 17 MPa (x SF 1.5) | 415 MPa | AN2 | 22 MPa | 0.30 | ok |
| marm247 | 1088 K | 601 + 17 MPa (x SF 1.5) | 557 MPa | AN2 | 22 MPa | 0.22 | above_material_data |

- selected material: NONE (infeasible)
- impeller tip speed 482 m/s (limit 500)
- AN^2 = 4.03e+07 m2 rpm2
- feasible: False (active constraints: AN2, TIT_ngv_above_material_data)
- material status: fails_creep
- turbine design-range flags (DePaola 2020 Table 2, sanity only): turbine an2_m2_rpm2 = 4.03e+07 outside DePaola Table 2 range [None, 32258000.0]; turbine loading_coefficient = 1 outside DePaola Table 2 range [1.4, 2.0]
- tags: turbine efficiency at micro scale unvalidated (Kofskey validation turbine is 0.24 m), turbine an2_m2_rpm2 = 4.03e+07 outside DePaola Table 2 range [None, 32258000.0], turbine loading_coefficient = 1 outside DePaola Table 2 range [1.4, 2.0]

## Off-design (physics-evaluated, speed-controlled)

| point | N [rpm] | alt [m] | Mach | Fn [N] | TSFC [g/(kN s)] | m_air [kg/s] | PR_c | TIT [K] | EGT [K] | status |
|---|---|---|---|---|---|---|---|---|---|---|
| design_check | 130000 | 0 | 0.00 | 148.3 | 41.5 | 0.300 | 3.000 | 1150 | 1033 | ok |
| cruise_M05_3km | 130000 | 3000 | 0.50 | 99.5 | 48.4 | 0.257 | 3.052 | 1087 | 968 | ok |
| part_speed_80 | 104000 | 0 | 0.00 | 72.3 | 52.0 | 0.216 | 2.047 | 1007 | 931 | ok |
| part_speed_60 | 78000 | 0 | 0.00 | 33.3 | 73.8 | 0.146 | 1.503 | 954 | 912 | ok |
| idle_35 | 45500 | 0 | 0.00 | 9.5 | 139.1 | 0.077 | 1.150 | 942 | 927 | ok |

Component assumptions: configs/engine_baseline.yaml (DePaola 2020 values as summarised by the user; compressor calibration from the KJ-66, provisional).

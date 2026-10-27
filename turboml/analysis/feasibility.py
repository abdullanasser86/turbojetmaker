"""Why is a design infeasible: PHYSICS or MISSING DATA? (user request 2026-10-27, round 5 item 3b)

The creep check (turboml.physics.creep) gives each blade row of each material one status:
  ok | AN2 | creep (rotor) and ok | creep (NGV)   -> decided WITH data: the stress x SF exceeds the 100 h rupture
                                                     strength at that metal temperature (physics, data in range);
  above_material_data                             -> the metal temperature is above the highest tabulated
                                                     temperature, so the check cannot be made (missing data).
A design is classified per material, then over the materials the model may select (default, premium):

  physics       every material fails at least one check that the data cover (AN2, creep_rotor, creep_ngv), or the
                IN-713C metal temperature is above its solidus (Nickel 337 p. 4: melting range 2300-2350 F,
                1533-1561 K; an uncooled blade at that gas temperature melts, so no rupture data can help);
  missing_data  at least one material passes every check its data cover and fails only because a metal
                temperature is above its data (more rupture data COULD change the verdict);
  other         infeasible for a non-creep reason only (e.g. impeller tip-speed limit).

More rupture data cannot open a design classified "physics". Assumptions: uncooled blades (metal temperature = gas
temperature as modelled in turboml.physics.creep), the configured pattern factor, life and safety factor.
"""

from __future__ import annotations

import pandas as pd

IN713C_SOLIDUS_K = (
    1533.15  # Nickel 337 p. 4: 2300 F (1260 C), lower end of the melting range [typical]
)
WITHIN_DATA_FAIL_ROTOR = ("AN2", "creep")
WITHIN_DATA_FAIL_NGV = ("creep",)


def classify_row(
    r: pd.Series, materials=("in713c", "marm247"), pattern_factor: float = 0.25
) -> dict:
    """Cause of infeasibility for one flattened dataset row (columns creep_<mat>_* and active_constraints)."""
    t_ngv = (
        r["in_tit_K"] + pattern_factor * (r["in_tit_K"] - r["T03_K"])
        if "T03_K" in r
        else float("nan")
    )
    per = {}
    for m in materials:
        rs, ns = r.get(f"creep_{m}_rotor_status"), r.get(f"creep_{m}_ngv_status")
        if not isinstance(rs, str):
            continue
        phys = []
        if rs in WITHIN_DATA_FAIL_ROTOR:
            phys.append("AN2" if rs == "AN2" else "creep_rotor")
        if ns in WITHIN_DATA_FAIL_NGV:
            phys.append("creep_ngv")
        if m == "in713c" and t_ngv > IN713C_SOLIDUS_K:
            phys.append("ngv_above_solidus")
        if m == "in713c" and r.get(f"creep_{m}_rotor_metal_T_K", 0.0) > IN713C_SOLIDUS_K:
            phys.append("rotor_above_solidus")
        data = [
            f"{row}_above_data"
            for row, st in (("rotor", rs), ("ngv", ns))
            if st == "above_material_data"
        ]
        per[m] = {"physics": phys, "missing_data": data}
    creep_ok = any(not v["physics"] and not v["missing_data"] for v in per.values())
    other = [c for c in str(r.get("active_constraints", "") or "").split(";")
             if c and not c.startswith(("AN2", "creep_", "TIT_"))]  # fmt: skip
    if creep_ok:
        cause = "other" if other else "feasible_creep"
    elif all(v["physics"] for v in per.values()):
        cause = "physics"
    else:
        cause = "missing_data"
    data_only = [m for m, v in per.items() if not v["physics"] and v["missing_data"]]
    return {"cause": cause, "ngv_metal_T_K": t_ngv, "data_limited_materials": ";".join(data_only),
            **{f"{m}_physics": ";".join(v["physics"]) for m, v in per.items()},
            **{f"{m}_missing_data": ";".join(v["missing_data"]) for m, v in per.items()}}  # fmt: skip


def classify(
    designs: pd.DataFrame, design_points: pd.DataFrame | None = None, **kw
) -> pd.DataFrame:
    """Classify every infeasible, designed engine. `design_points`: points.csv rows with point == 'design'
    (supplies T03_K for the NGV hot-streak temperature)."""
    d = designs[(designs.status == "ok") & ~designs.feasible.astype(bool)].copy()
    if design_points is not None:
        d = d.merge(design_points[["design_id", "T03_K"]], on="design_id", how="left")
    out = pd.DataFrame([classify_row(r, **kw) for _, r in d.iterrows()], index=d.index)
    return pd.concat([d, out], axis=1)

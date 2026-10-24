"""International Standard Atmosphere (ISA, = US Standard Atmosphere 1976 below 20 km).

The atmosphere sets the air conditions the engine breathes. ISA is defined in
*geopotential* altitude H (which treats gravity as constant):

* troposphere, 0–11 km:     T = 288.15 K − 6.5 K/km · H,   p = p0 (T/T0)^(g0 / (R L))
* lower stratosphere, 11–20 km: T = 216.65 K (constant),  p = p11 · exp(−g0 (H − 11 km) / (R T))

Geometric altitude h (what an altimeter-on-a-map means) converts as H = r0 h / (r0 + h),
r0 = 6 356 766 m. The difference is 1.4 m at 3 km — negligible for us, but done properly.

Optional temperature offset dT_isa shifts T but not p (hot/cold day convention used by
pyCycle's FlightConditions ``dTs`` input).

Source: U.S. Standard Atmosphere, 1976 (NOAA/NASA/USAF), NASA-TM-X-74335.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from turboml.units import G0, ISA_SEA_LEVEL_P_PA, ISA_SEA_LEVEL_T_K

R_AIR_ISA = 287.05287  # J/(kg K), value used to define ISA
R0_EARTH_M = 6356766.0
LAPSE_K_PER_M = 0.0065
H_TROPOPAUSE_M = 11000.0
T_TROPOPAUSE_K = 216.65


@dataclass(frozen=True)
class AmbientState:
    T_K: float
    p_Pa: float


def geopotential_m(alt_geometric_m: float) -> float:
    """Geometric altitude [m] -> geopotential altitude [m]."""
    return R0_EARTH_M * alt_geometric_m / (R0_EARTH_M + alt_geometric_m)


def isa(alt_geometric_m: float, dT_isa_K: float = 0.0) -> AmbientState:
    """Static temperature and pressure at geometric altitude [m], valid −0.5 to 20 km."""
    if not -500.0 <= alt_geometric_m <= 20000.0:
        raise ValueError("ISA model implemented for -500 m ... 20 km only")
    H = geopotential_m(alt_geometric_m)
    expo = G0 / (R_AIR_ISA * LAPSE_K_PER_M)
    if H <= H_TROPOPAUSE_M:
        T = ISA_SEA_LEVEL_T_K - LAPSE_K_PER_M * H
        p = ISA_SEA_LEVEL_P_PA * (T / ISA_SEA_LEVEL_T_K) ** expo
    else:
        p11 = ISA_SEA_LEVEL_P_PA * (T_TROPOPAUSE_K / ISA_SEA_LEVEL_T_K) ** expo
        T = T_TROPOPAUSE_K
        p = p11 * math.exp(-G0 * (H - H_TROPOPAUSE_M) / (R_AIR_ISA * T))
    return AmbientState(T_K=T + dT_isa_K, p_Pa=p)

"""Unit conventions and conversions.

TurboML uses **SI units internally, everywhere**: m, kg, s, K, Pa, N, W, J/kg, rad/s.
Variables whose unit is not obvious carry a suffix, e.g. ``n_rpm`` vs ``omega_rad_s``,
``tsfc_kg_per_N_s`` vs ``tsfc_g_per_kN_s``.

pyCycle works in US customary units by default (psi, degR, lbm/s, lbf, Btu), so every
value crossing the pyCycle boundary goes through the converters below. Conversion factors
are exact definitions (NIST SP 811) unless noted.
"""

from __future__ import annotations

import math

# --- exact definitions -------------------------------------------------------
LBM_TO_KG = 0.45359237
LBF_TO_N = 4.4482216152605  # lbf = lbm * g0
PSI_TO_PA = 6894.757293168361  # lbf / in^2
INCH_TO_M = 0.0254
FT_TO_M = 0.3048
RANKINE_TO_KELVIN = 5.0 / 9.0
BTU_TO_J = 1055.05585262  # International Table Btu
HP_TO_W = 745.69987158227022  # mechanical horsepower

# --- standard constants --------------------------------------------------------
G0 = 9.80665  # m/s^2, standard gravity
ISA_SEA_LEVEL_T_K = 288.15
ISA_SEA_LEVEL_P_PA = 101325.0


def rpm_to_rad_s(n_rpm: float) -> float:
    """Shaft speed: revolutions per minute -> angular velocity [rad/s]."""
    return n_rpm * 2.0 * math.pi / 60.0


def rad_s_to_rpm(omega_rad_s: float) -> float:
    """Angular velocity [rad/s] -> revolutions per minute."""
    return omega_rad_s * 60.0 / (2.0 * math.pi)


def degR_to_K(t_degR: float) -> float:
    return t_degR * RANKINE_TO_KELVIN


def K_to_degR(t_K: float) -> float:
    return t_K / RANKINE_TO_KELVIN


def psi_to_Pa(p_psi: float) -> float:
    return p_psi * PSI_TO_PA


def Pa_to_psi(p_Pa: float) -> float:
    return p_Pa / PSI_TO_PA


def lbm_s_to_kg_s(w_lbm_s: float) -> float:
    return w_lbm_s * LBM_TO_KG


def kg_s_to_lbm_s(w_kg_s: float) -> float:
    return w_kg_s / LBM_TO_KG


def lbf_to_N(f_lbf: float) -> float:
    return f_lbf * LBF_TO_N


def N_to_lbf(f_N: float) -> float:
    return f_N / LBF_TO_N


def tsfc_lbm_per_lbf_h_to_kg_per_N_s(tsfc: float) -> float:
    """TSFC [lbm/(lbf*h)] -> [kg/(N*s)].

    TSFC (thrust-specific fuel consumption) = fuel mass flow / thrust.
    """
    return tsfc * LBM_TO_KG / (LBF_TO_N * 3600.0)


def tsfc_kg_per_N_s_to_g_per_kN_s(tsfc: float) -> float:
    """[kg/(N*s)] -> [g/(kN*s)], the unit most micro-turbojet literature uses."""
    return tsfc * 1e6


def tsfc_kg_per_N_s_to_kg_per_N_h(tsfc: float) -> float:
    """[kg/(N*s)] -> [kg/(N*h)], the unit many hobby-engine datasheets use."""
    return tsfc * 3600.0

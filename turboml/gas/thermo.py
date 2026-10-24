"""Ideal-gas thermodynamic properties of air and kerosene combustion products.

Physics in plain language
-------------------------
A gas turbine's working fluid is air before the combustor and a mixture of N2, O2, Ar,
CO2 and H2O after it. We treat each species as an *ideal gas* (pressure = rho R T),
which is accurate at engine pressures (< 10 bar) and temperatures (200–2000 K). But
the specific heat cp of every species rises with temperature, so we do **not** assume
constant cp ("calorically perfect gas"). That is the "variable cp" model.

For each species we use the NASA 9-coefficient polynomials (McBride, Zehe & Gordon,
NASA TP-2002-211556) in their standard form::

    cp/R  = a1 T^-2 + a2 T^-1 + a3 + a4 T + a5 T^2 + a6 T^3 + a7 T^4
    H/RT  = -a1 T^-2 + a2 ln(T)/T + a3 + a4 T/2 + a5 T^2/3 + a6 T^3/4 + a7 T^4/5 + b1/T
    S°/R  = -a1 T^-2/2 - a2/T + a3 ln T + a4 T + a5 T^2/2 + a6 T^3/3 + a7 T^4/4 + b2

H is the *absolute* enthalpy: it includes the enthalpy of formation (elements in their
reference state at 298.15 K have H = 0). That is what lets a single energy balance
handle combustion: the chemical energy of the fuel shows up as the difference between
the formation enthalpies of reactants and products.

For a mixture with fixed composition (n_i mol per kg), h = Σ n_i H_i and
s(T, p) = Σ n_i S°_i(T) − R_mix ln(p/p_ref) + const. An isentropic process at fixed
composition therefore satisfies  Σ n_i S°_i(T2) − Σ n_i S°_i(T1) = R_mix ln(p2/p1).

Combustion is modelled as **complete** (all C → CO2, all H → H2O). No dissociation, no
CO / unburned hydrocarbons. Dissociation is negligible below ~1500 K, which covers
uncooled micro-turbojet turbine inlet temperatures. Inefficiency of real combustors is
handled separately by a combustion efficiency (see :func:`fuel_enthalpy_J_per_kg`).

Data source
-----------
The coefficients are read from pyCycle's copy of the NASA/CEA data
(``pycycle.thermo.cea.thermo_data.janaf``). Using *the same property data* as pyCycle is
deliberate. It lets the two cycle models be compared with matched assumptions, so any
disagreement points at a cycle-modelling bug, not at property data. Only the data is
shared; every equation and solver here is independent of pyCycle.

Atomic weights are taken consistent with the species molar masses (C = 12.0107).
pyCycle's own element table lists C = 12.017, a ~0.05 % typo; see docs/DECISIONS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pycycle.thermo.cea.thermo_data import janaf

R_UNIVERSAL = 8.314462618  # J/(mol K), CODATA 2018 exact
T_REF_K = 298.15
P_REF_PA = 101325.0

SPECIES = ("N2", "O2", "Ar", "CO2", "H2O")

# Atomic weights consistent with the species molar masses in the data set [g/mol].
ATOMIC_WT = {"N": 28.01348 / 2, "O": 31.9988 / 2, "Ar": 39.948, "H": 1.00794}
ATOMIC_WT["C"] = 44.0095 - 2 * ATOMIC_WT["O"]  # = 12.0107

# Dry air elemental composition, mol of element per gram of air, exactly as pyCycle's
# CEA_AIR_COMPOSITION (pycycle/constants.py). Implies ~0.04 % CO2 by volume.
AIR_ELEMENTS_MOL_PER_G = {
    "N": 5.39157698e-02,
    "O": 1.44860137e-02,
    "Ar": 3.23319235e-04,
    "C": 1.10132233e-05,
}

# Kerosene surrogate: Jet-A as C12H23, the same formula as pyCycle's 'Jet-A(g)'.
FUEL_C, FUEL_H = 12.0, 23.0
FUEL_MW_G_MOL = FUEL_C * ATOMIC_WT["C"] + FUEL_H * ATOMIC_WT["H"]


@dataclass(frozen=True)
class _Species:
    mw: float  # g/mol
    t_breaks: np.ndarray  # interval edges [K]
    coeffs: np.ndarray  # (n_intervals, 9)


def _load_species() -> dict[str, _Species]:
    out = {}
    for name in SPECIES:
        d = janaf.products[name]
        coeffs = np.asarray(d["coeffs"], dtype=float)
        breaks = np.asarray(d["ranges"], dtype=float)[: coeffs.shape[0] + 1]
        out[name] = _Species(mw=float(d["wt"]), t_breaks=breaks, coeffs=coeffs)
    return out


_SP = _load_species()


def _coeffs_at(sp: _Species, T: float) -> np.ndarray:
    if not (sp.t_breaks[0] <= T <= sp.t_breaks[-1]):
        raise ValueError(f"T = {T:.1f} K outside NASA polynomial range {sp.t_breaks[[0, -1]]}")
    i = min(int(np.searchsorted(sp.t_breaks, T, side="right")) - 1, len(sp.coeffs) - 1)
    return sp.coeffs[i]


def species_cp_h_s(name: str, T: float) -> tuple[float, float, float]:
    """Molar cp [J/mol/K], absolute H [J/mol] and S° at 1 atm [J/mol/K] of one species."""
    a = _coeffs_at(_SP[name], T)
    lnT = np.log(T)
    cp = a[0] / T**2 + a[1] / T + a[2] + a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4
    h = (
        -a[0] / T**2
        + a[1] * lnT / T
        + a[2]
        + a[3] * T / 2
        + a[4] * T**2 / 3
        + a[5] * T**3 / 4
        + a[6] * T**4 / 5
        + a[7] / T
    )
    s = (
        -a[0] / (2 * T**2)
        - a[1] / T
        + a[2] * lnT
        + a[3] * T
        + a[4] * T**2 / 2
        + a[5] * T**3 / 3
        + a[6] * T**4 / 4
        + a[8]
    )
    return cp * R_UNIVERSAL, h * R_UNIVERSAL * T, s * R_UNIVERSAL


class GasMixture:
    """Ideal-gas mixture of fixed composition, defined by mol of each species per kg.

    All properties are per kg of mixture, SI units. Enthalpy is absolute (includes
    formation enthalpy); entropy uses the 1 atm standard-state reference.

    Speed: because every property is linear in the NASA coefficients, the mixture's
    coefficients are summed once per temperature interval (all species used here share the
    200/1000/6000 K breaks). Each property is then a single polynomial evaluation, which is
    mathematically identical to summing species by species.
    """

    T_MIN, T_BREAK, T_MAX = 200.0, 1000.0, 6000.0

    def __init__(self, mol_per_kg: dict[str, float]):
        self.n = {k: float(v) for k, v in mol_per_kg.items() if v != 0.0}
        mass = sum(v * _SP[k].mw for k, v in self.n.items()) / 1000.0
        if abs(mass - 1.0) > 1e-9:
            raise ValueError(f"composition does not sum to 1 kg (got {mass:.12f} kg)")
        self.R = R_UNIVERSAL * sum(self.n.values())  # J/(kg K)
        for k in self.n:
            if not np.allclose(_SP[k].t_breaks[:3], [self.T_MIN, self.T_BREAK, self.T_MAX]):
                raise ValueError(f"species {k} has different temperature breaks")
        # mixture coefficients (per kg) for the two intervals, as plain floats
        self._a = [
            [
                R_UNIVERSAL * sum(n * _SP[k].coeffs[i][j] for k, n in self.n.items())
                for j in range(9)
            ]
            for i in (0, 1)
        ]

    def _coef(self, T: float):
        if not (self.T_MIN <= T <= self.T_MAX):
            raise ValueError(f"T = {T:.1f} K outside NASA polynomial range [200, 6000] K")
        return self._a[0] if T <= self.T_BREAK else self._a[1]

    def cp(self, T: float) -> float:
        a = self._coef(T)
        return a[0] / (T * T) + a[1] / T + a[2] + T * (a[3] + T * (a[4] + T * (a[5] + T * a[6])))

    def h(self, T: float) -> float:
        a = self._coef(T)
        return (
            -a[0] / T + a[1] * math.log(T) + T * (a[2] + T * (a[3] / 2 + T * (a[4] / 3 + T * (a[5] / 4 + T * a[6] / 5))))
            + a[7]
        )  # fmt: skip

    def s0(self, T: float) -> float:
        """Σ n_i S°_i(T): the temperature part of the entropy [J/(kg K)]."""
        a = self._coef(T)
        return (
            -a[0] / (2 * T * T) - a[1] / T + a[2] * math.log(T)
            + T * (a[3] + T * (a[4] / 2 + T * (a[5] / 3 + T * a[6] / 4))) + a[8]
        )  # fmt: skip

    def gamma(self, T: float) -> float:
        cp = self.cp(T)
        return cp / (cp - self.R)

    def sound_speed(self, T: float) -> float:
        return math.sqrt(self.gamma(T) * self.R * T)

    # --- inversions (Newton; the functions are smooth and monotonic) ---------------
    def T_from_h(self, h: float, T_guess: float = 800.0) -> float:
        T = T_guess
        for _ in range(50):
            dT = (self.h(T) - h) / self.cp(T)
            T -= dT
            if abs(dT) < 1e-9 * T:
                return T
        raise RuntimeError(f"T_from_h did not converge (h = {h:.6g} J/kg)")

    def T_isentropic(self, T1: float, p_ratio: float) -> float:
        """Temperature after an isentropic change p2/p1 = p_ratio from T1."""
        target = self.s0(T1) + self.R * math.log(p_ratio)
        T = T1 * p_ratio ** (self.R / self.cp(T1))  # constant-cp first guess
        for _ in range(50):
            dT = (self.s0(T) - target) / (self.cp(T) / T)
            T -= dT
            if abs(dT) < 1e-9 * T:
                return T
        raise RuntimeError("T_isentropic did not converge")

    def p_ratio_isentropic(self, T1: float, T2: float) -> float:
        """p2/p1 of an isentropic process between T1 and T2."""
        return math.exp((self.s0(T2) - self.s0(T1)) / self.R)


@lru_cache(maxsize=1)
def air() -> GasMixture:
    """Dry air with pyCycle's CEA elemental composition, as N2/O2/Ar/CO2 species."""
    e = AIR_ELEMENTS_MOL_PER_G
    n = {
        "N2": e["N"] / 2,
        "CO2": e["C"],
        "O2": (e["O"] - 2 * e["C"]) / 2,
        "Ar": e["Ar"],
    }
    return _normalise({k: v * 1000.0 for k, v in n.items()})


def products(far: float) -> GasMixture:
    """Complete-combustion products of 1 kg air + ``far`` kg Jet-A (C12H23), per kg mix.

    Each mol of fuel consumes 12 + 23/4 = 17.75 mol O2 and yields 12 CO2 + 11.5 H2O.
    """
    if far < 0:
        raise ValueError("FAR must be >= 0")
    a = air().n  # mol per kg air
    n_fuel = far * 1000.0 / FUEL_MW_G_MOL
    o2 = a["O2"] - n_fuel * (FUEL_C + FUEL_H / 4)
    if o2 < 0:
        raise ValueError(f"FAR = {far:.4f} is richer than stoichiometric")
    n = {
        "N2": a["N2"],
        "Ar": a["Ar"],
        "O2": o2,
        "CO2": a["CO2"] + n_fuel * FUEL_C,
        "H2O": n_fuel * FUEL_H / 2,
    }
    return _normalise(n, total_mass_kg=1.0 + far)


def stoichiometric_far() -> float:
    return air().n["O2"] / (FUEL_C + FUEL_H / 4) * FUEL_MW_G_MOL / 1000.0


def _normalise(n: dict[str, float], total_mass_kg: float | None = None) -> GasMixture:
    mass = sum(v * _SP[k].mw for k, v in n.items()) / 1000.0
    if total_mass_kg is not None and abs(mass - total_mass_kg) > 1e-6 * total_mass_kg:
        raise AssertionError(f"mass not conserved: {mass} vs {total_mass_kg}")
    return GasMixture({k: v / mass for k, v in n.items()})


def fuel_enthalpy_J_per_kg(lhv_J_per_kg: float, eta_burner: float = 1.0) -> float:
    """Effective absolute enthalpy of the liquid fuel entering the combustor [J/kg fuel].

    The lower heating value (LHV) is the heat released when 1 kg of fuel burns
    completely and the water stays as vapour, with reactants and products at 298.15 K::

        LHV = h_fuel − (12 H_CO2 + 11.5 H_H2O − 17.75 H_O2)(298.15 K) / M_fuel

    so  h_fuel = Σ(products − O2) / M_fuel + LHV.  Defining the fuel by its LHV (a measured,
    specified property; ASTM D1655 sets a minimum for Jet A/A-1) rather than by a
    heat of formation keeps the energy balance tied to the real fuel.

    Combustion efficiency η_b (fraction of the LHV actually released) is applied by
    lowering the effective fuel enthalpy by (1 − η_b)·LHV. That is the textbook
    definition η_b = actual / ideal heat release. The products are still treated as
    complete-combustion species (the unburned fraction's tiny composition effect is
    neglected).

    The fuel enters at 298.15 K; its sensible enthalpy relative to that is neglected.
    """
    T = T_REF_K
    h_prod = (
        FUEL_C * species_cp_h_s("CO2", T)[1]
        + FUEL_H / 2 * species_cp_h_s("H2O", T)[1]
        - (FUEL_C + FUEL_H / 4) * species_cp_h_s("O2", T)[1]
    )
    return h_prod / (FUEL_MW_G_MOL / 1000.0) + eta_burner * lhv_J_per_kg

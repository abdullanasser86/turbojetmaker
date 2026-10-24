"""Gas property model: standard reference values and conservation laws."""

import pytest

from turboml.gas import thermo as th


@pytest.mark.parametrize(
    "species, cp_298, s_298, hf_298",
    [
        # JANAF / NIST-JANAF thermochemical tables, 298.15 K, 1 bar-ish standard state
        ("N2", 29.124, 191.61, 0.0),
        ("O2", 29.376, 205.15, 0.0),
        ("Ar", 20.786, 154.85, 0.0),
        ("CO2", 37.135, 213.79, -393510.0),
        ("H2O", 33.590, 188.84, -241826.0),
    ],
)
def test_species_reference_values(species, cp_298, s_298, hf_298):
    cp, h, s = th.species_cp_h_s(species, 298.15)
    assert cp == pytest.approx(cp_298, abs=0.01)
    assert s == pytest.approx(s_298, abs=0.05)
    assert h == pytest.approx(hf_298, abs=20.0)


def test_air_properties():
    a = th.air()
    assert a.R == pytest.approx(287.05, abs=0.02)
    assert a.gamma(300.0) == pytest.approx(1.400, abs=0.001)
    assert a.cp(300.0) == pytest.approx(1005.0, abs=1.0)
    assert a.cp(1000.0) > a.cp(300.0)  # variable cp


def test_products_conserve_mass_and_atoms():
    far = 0.02
    p = th.products(far)  # per kg of mixture; mass check is inside _normalise
    n_fuel_per_kg_mix = far * 1000 / th.FUEL_MW_G_MOL / (1 + far)
    co2_from_air = th.air().n["CO2"] / (1 + far)
    assert p.n["H2O"] == pytest.approx(11.5 * n_fuel_per_kg_mix, rel=1e-12)
    assert p.n["CO2"] - co2_from_air == pytest.approx(12 * n_fuel_per_kg_mix, rel=1e-12)


def test_stoichiometric_far_kerosene():
    # Kerosene stoichiometric FAR is ~0.068 (air/fuel ~14.7)
    assert th.stoichiometric_far() == pytest.approx(0.0682, abs=0.0005)
    with pytest.raises(ValueError):
        th.products(0.08)


def test_fuel_enthalpy_reproduces_lhv():
    """Burning 1 kg fuel at 298.15 K with stoichiometric O2 must release exactly the LHV."""
    lhv = 42.8e6
    T = th.T_REF_K
    n_f = 1000.0 / th.FUEL_MW_G_MOL  # mol fuel per kg
    h_react = th.fuel_enthalpy_J_per_kg(lhv) + n_f * 17.75 * th.species_cp_h_s("O2", T)[1]
    h_prod = n_f * (12 * th.species_cp_h_s("CO2", T)[1] + 11.5 * th.species_cp_h_s("H2O", T)[1])
    assert h_react - h_prod == pytest.approx(lhv, rel=1e-12)
    # combustion efficiency removes (1 - eta) * LHV
    assert th.fuel_enthalpy_J_per_kg(lhv, 0.9) == pytest.approx(
        th.fuel_enthalpy_J_per_kg(lhv) - 0.1 * lhv, rel=1e-12
    )


def test_inversions_round_trip():
    g = th.products(0.015)
    for T in (300.0, 999.0, 1001.0, 1400.0):  # both sides of the 1000 K polynomial break
        assert g.T_from_h(g.h(T)) == pytest.approx(T, rel=1e-10)
        T2 = g.T_isentropic(T, 2.5)
        assert g.p_ratio_isentropic(T, T2) == pytest.approx(2.5, rel=1e-10)

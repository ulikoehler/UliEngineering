#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kohlrausch's law of independent migration of ions.

Kohlrausch's law states that the limiting molar conductivity of an electrolyte.
is the sum of the individual contributions of the cation and anion:

    Λ°_m = ν+ * λ°+ + ν- * λ°-

The concentration dependence:
    Λ_m = Λ°_m - K * √c   (Kohlrausch's square root law)

where K is the Kohlrausch coefficient.
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
import numpy as np

__all__ = [
    "kohlrausch_limiting_molar_conductivity",
    "kohlrausch_molar_conductivity",
    "kohlrausch_coefficient_from_data",
    "transference_number",
    "LIMITING_MOLAR_CONDUCTIVITIES",
    "normalize_molar_conductivity", "MolarConductivitySCm2Mol",
    "normalize_concentration", "ConcentrationMolar",
]


def normalize_molar_conductivity(lambda_val: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(lambda_val, {"S·cm²/mol": 1.0, "S cm2/mol": 1.0, "S·m²/mol": 10000.0, "S m2/mol": 10000.0}, quantity_name="molar conductivity")

def normalize_concentration(c: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(c, {"mol/L": 1.0, "M": 1.0, "mM": 1e-3, "µM": 1e-6, "mol/m³": 1e-3}, quantity_name="concentration")

MolarConductivitySCm2Mol = Annotated[NormalizedComputable, normalize_molar_conductivity]
ConcentrationMolar = Annotated[NormalizedComputable, normalize_concentration]

# Limiting molar conductivities at 25 °C in S·cm²/mol
# Source: CRC Handbook of Chemistry and Physics
LIMITING_MOLAR_CONDUCTIVITIES = {
    # Cations
    "H+": 349.81,
    "Li+": 38.66,
    "Na+": 50.10,
    "K+": 73.50,
    "Rb+": 77.81,
    "Cs+": 77.26,
    "Ag+": 61.90,
    "NH4+": 73.55,
    "Mg2+": 106.12,
    "Ca2+": 119.00,
    "Ba2+": 127.28,
    "Cu2+": 107.2,
    "Zn2+": 105.6,
    "Fe2+": 108.0,
    "Fe3+": 204.0,
    "Al3+": 189.0,
    # Anions
    "OH-": 198.3,
    "F-": 55.4,
    "Cl-": 76.35,
    "Br-": 78.14,
    "I-": 76.84,
    "NO3-": 71.46,
    "ClO4-": 67.36,
    "CH3COO-": 40.9,
    "HCO3-": 44.5,
    "SO4_2-": 160.0,
    "CO3_2-": 138.6,
    "PO4_3-": 207.0,
}


@returns_unit("S·cm²/mol")
def kohlrausch_limiting_molar_conductivity(lambda_ions, stoich_coefficients):
    """Compute the limiting molar conductivity of an electrolyte
    
    using Kohlrausch's law of independent migration.

    Λ°_m = Σ(ν_i * λ°_i)

    Parameters
    ----------
    lambda_ions : array-like
        Limiting molar conductivities of individual ions in S·cm²/mol.
    stoich_coefficients : array-like
        Stoichiometric coefficients for each ion.

    Returns
    -------
    float
        Limiting molar conductivity in S·cm²/mol.
    
    """
    lam = np.asarray(lambda_ions, dtype=float)
    nu = np.asarray(stoich_coefficients, dtype=float)
    return float(np.sum(nu * lam))


@returns_unit("S·cm²/mol")
def kohlrausch_molar_conductivity(Lambda_0: MolarConductivitySCm2Mol, K, c: ConcentrationMolar):
    """Compute molar conductivity at concentration c using Kohlrausch's square root law.

    Λ_m = Λ°_m - K * √c.

    Parameters
    ----------
    Lambda_0 : float
        Limiting molar conductivity in S·cm²/mol.
    K : float
        Kohlrausch coefficient in S·cm²/(mol^(3/2)·L^(1/2)).
    c : float
        Concentration in mol/L.

    Returns
    -------
    float
        Molar conductivity in S·cm²/mol.
    
    """
    Lambda_0 = normalize_molar_conductivity(Lambda_0) if isinstance(Lambda_0, str) else Lambda_0
    c = normalize_concentration(c) if isinstance(c, str) else c
    return Lambda_0 - K * np.sqrt(c)


@returns_unit("S·cm²/(mol^(3/2)·L^(1/2))")
def kohlrausch_coefficient_from_data(Lambda_0: MolarConductivitySCm2Mol, Lambda_m: MolarConductivitySCm2Mol, c: ConcentrationMolar):
    """Determine the Kohlrausch coefficient K from experimental data.

    K = (Λ°_m - Λ_m) / √c.

    Parameters
    ----------
    Lambda_0 : float
        Limiting molar conductivity in S·cm²/mol.
    Lambda_m : float
        Measured molar conductivity at concentration c.
    c : float
        Concentration in mol/L.

    Returns
    -------
    float
        Kohlrausch coefficient K.
    
    """
    Lambda_0 = normalize_molar_conductivity(Lambda_0) if isinstance(Lambda_0, str) else Lambda_0
    Lambda_m = normalize_molar_conductivity(Lambda_m) if isinstance(Lambda_m, str) else Lambda_m
    c = normalize_concentration(c) if isinstance(c, str) else c
    return (Lambda_0 - Lambda_m) / np.sqrt(c)


@returns_unit("")
def transference_number(lambda_ion: MolarConductivitySCm2Mol, Lambda_0: MolarConductivitySCm2Mol):
    """Compute the transference number (transport number) of an ion.

    t_i = λ_i / Λ°_m.

    Parameters
    ----------
    lambda_ion : float
        Limiting molar conductivity of the ion in S·cm²/mol.
    Lambda_0 : float
        Limiting molar conductivity of the electrolyte in S·cm²/mol.

    Returns
    -------
    float
        Transference number (dimensionless, between 0 and 1).
    
    """
    lambda_ion = normalize_molar_conductivity(lambda_ion) if isinstance(lambda_ion, str) else lambda_ion
    Lambda_0 = normalize_molar_conductivity(Lambda_0) if isinstance(Lambda_0, str) else Lambda_0
    return lambda_ion / Lambda_0

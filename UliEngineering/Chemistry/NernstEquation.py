#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nernst equation for electrochemical cell potential.

The Nernst equation relates the reduction potential of an electrochemical.
reaction to the standard electrode potential, temperature, and activities
(or concentrations) of the chemical species undergoing reduction and oxidation.

E = E0 - (R*T)/(n*F) * ln(Q)

where:
    E0 = standard cell potential (V)
    R  = gas constant (8.314 J/(mol·K))
    T  = temperature (K)
    n  = number of electrons transferred
    F  = Faraday constant (96485 C/mol)
    Q  = reaction quotient
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature
import numpy as np
from scipy.constants import R as gas_constant, physical_constants

__all__ = [
    "nernst_cell_potential",
    "nernst_half_cell_potential",
    "nernst_potential_at_25C",
    "nernst_reaction_quotient_from_potential",
    "FARADAY_CONSTANT",
    "normalize_concentration", "ConcentrationMolar",
]

FARADAY_CONSTANT = physical_constants["Faraday constant"][0]  # 96485.33212 C/mol


def normalize_concentration(c: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(c, {"mol/L": 1.0, "M": 1.0, "mM": 1e-3, "µM": 1e-6, "mol/m³": 1e-3}, quantity_name="concentration")

ConcentrationMolar = Annotated[NormalizedComputable, normalize_concentration]


@returns_unit("V")
def nernst_cell_potential(E0, n, Q, T=298.15):
    """
    Compute the cell potential using the Nernst equation.

    E = E0 - (R*T)/(n*F) * ln(Q).

    Parameters
    ----------
    E0 : float
        Standard cell potential in Volts.
    n : float
        Number of electrons transferred in the reaction.
    Q : float
        Reaction quotient (ratio of product activities to reactant activities).
    T : float
        Temperature in Kelvin (default: 298.15 K = 25 °C).

    Returns
    -------
    float
        Cell potential in Volts.

    """
    T = normalize_temperature(T) if isinstance(T, str) else T
    return E0 - (gas_constant * T) / (n * FARADAY_CONSTANT) * np.log(Q)


@returns_unit("V")
def nernst_half_cell_potential(E0, n, oxidized_concentration: ConcentrationMolar, reduced_concentration: ConcentrationMolar, T=298.15):
    """
    Compute the half-cell reduction potential using the Nernst equation.

    E = E0 - (R*T)/(n*F) * ln([Red]/[Ox]).

    Parameters
    ----------
    E0 : float
        Standard reduction potential in Volts.
    n : float
        Number of electrons transferred.
    oxidized_concentration : float
        Concentration of the oxidized species (mol/L).
    reduced_concentration : float
        Concentration of the reduced species (mol/L).
    T : float
        Temperature in Kelvin (default: 298.15 K = 25 °C).

    Returns
    -------
    float
        Half-cell potential in Volts.

    """
    oxidized_concentration = normalize_concentration(oxidized_concentration) if isinstance(oxidized_concentration, str) else oxidized_concentration
    reduced_concentration = normalize_concentration(reduced_concentration) if isinstance(reduced_concentration, str) else reduced_concentration
    T = normalize_temperature(T) if isinstance(T, str) else T
    Q = reduced_concentration / oxidized_concentration
    return E0 - (gas_constant * T) / (n * FARADAY_CONSTANT) * np.log(Q)


@returns_unit("V")
def nernst_potential_at_25C(E0, n, Q):
    """
    Simplified Nernst equation at 25 °C (298.15 K).

    E = E0 - (0.02569 V / n) * ln(Q).
      = E0 - (0.05916 V / n) * log10(Q)

    Parameters
    ----------
    E0 : float
        Standard cell potential in Volts.
    n : float
        Number of electrons transferred.
    Q : float
        Reaction quotient.

    Returns
    -------
    float
        Cell potential in Volts at 25 °C.

    """
    return E0 - (0.025693 / n) * np.log(Q)


@returns_unit("")
def nernst_reaction_quotient_from_potential(E, E0, n, T=298.15):
    """
    Compute the reaction quotient Q from measured cell potential.
    
    using the inverse Nernst equation.

    Q = exp((E0 - E) * n * F / (R * T))

    Parameters
    ----------
    E : float
        Measured cell potential in Volts.
    E0 : float
        Standard cell potential in Volts.
    n : float
        Number of electrons transferred.
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Reaction quotient (dimensionless).

    """
    T = normalize_temperature(T) if isinstance(T, str) else T
    return np.exp((E0 - E) * n * FARADAY_CONSTANT / (gas_constant * T))

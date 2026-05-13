#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Henderson-Hasselbalch equation for pH/buffer calculations.

The Henderson-Hasselbalch equation relates pH, pKa, and the ratio.
of conjugate base to weak acid concentrations:

    pH = pKa + log10([A⁻] / [HA])

Also includes the Henderson equation for liquid junction potential:

    E_j = (R*T/F) * Σ((u_i - v_i)/(u_i + v_i)) * ln(a_i'' / a_i')
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature
import numpy as np
from scipy.constants import R as gas_constant, physical_constants

__all__ = [
    "henderson_hasselbalch_pH",
    "henderson_hasselbalch_ratio",
    "henderson_hasselbalch_pKa",
    "henderson_junction_potential",
    "henderson_junction_potential_simple",
    "buffer_capacity",
    "normalize_concentration", "ConcentrationMolar",
    "normalize_molar_conductivity", "MolarConductivitySM2Mol",
]


def normalize_concentration(c: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(c, {"mol/L": 1.0, "M": 1.0, "mM": 1e-3, "µM": 1e-6, "mol/m³": 1e-3}, quantity_name="concentration")

def normalize_molar_conductivity(lambda_val: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(lambda_val, {"S·m²/mol": 1.0, "S m2/mol": 1.0, "S·cm²/mol": 1e-4, "S cm2/mol": 1e-4}, quantity_name="molar conductivity")


ConcentrationMolar = Annotated[NormalizedComputable, normalize_concentration]
MolarConductivitySM2Mol = Annotated[NormalizedComputable, normalize_molar_conductivity]

FARADAY_CONSTANT = physical_constants["Faraday constant"][0]


@returns_unit("")
def henderson_hasselbalch_pH(pKa, base_concentration: ConcentrationMolar, acid_concentration: ConcentrationMolar):
    """
    Compute pH using the Henderson-Hasselbalch equation.

    pH = pKa + log10([A⁻] / [HA]).

    Parameters
    ----------
    pKa : float
        Acid dissociation constant (negative log).
    base_concentration : float
        Concentration of the conjugate base [A⁻] (mol/L).
    acid_concentration : float
        Concentration of the weak acid [HA] (mol/L).

    Returns
    -------
    float
        pH value (dimensionless).

    """
    base_concentration = normalize_concentration(base_concentration) if isinstance(base_concentration, str) else base_concentration
    acid_concentration = normalize_concentration(acid_concentration) if isinstance(acid_concentration, str) else acid_concentration
    return pKa + np.log10(base_concentration / acid_concentration)


@returns_unit("")
def henderson_hasselbalch_ratio(pH, pKa):
    """
    Compute the [A⁻]/[HA] ratio from pH and pKa.

    [A⁻] / [HA] = 10^(pH - pKa).

    Parameters
    ----------
    pH : float
        pH of the solution.
    pKa : float
        Acid dissociation constant.

    Returns
    -------
    float
        Ratio of conjugate base to acid concentration (dimensionless).
    
    """
    return 10.0 ** (pH - pKa)


@returns_unit("")
def henderson_hasselbalch_pKa(pH, base_concentration: ConcentrationMolar, acid_concentration: ConcentrationMolar):
    """
    Compute pKa from pH and concentrations using the Henderson-Hasselbalch equation.

    pKa = pH - log10([A⁻] / [HA]).

    Parameters
    ----------
    pH : float
        pH of the solution.
    base_concentration : float
        Concentration of conjugate base [A⁻] (mol/L).
    acid_concentration : float
        Concentration of weak acid [HA] (mol/L).

    Returns
    -------
    float
        pKa value (dimensionless).
    
    """
    base_concentration = normalize_concentration(base_concentration) if isinstance(base_concentration, str) else base_concentration
    acid_concentration = normalize_concentration(acid_concentration) if isinstance(acid_concentration, str) else acid_concentration
    return pH - np.log10(base_concentration / acid_concentration)


@returns_unit("V")
def henderson_junction_potential(t_plus, t_minus, c1: ConcentrationMolar, c2: ConcentrationMolar, T=298.15):
    """
    Compute the liquid junction potential using the Henderson equation.
    
    for a simple 1:1 electrolyte (e.g. KCl).

    E_j = (R*T/F) * (t+ - t-) * ln(c₂/c₁)

    where t+ and t- are the transference numbers of cation and anion.

    Parameters
    ----------
    t_plus : float
        Transference number of the cation.
    t_minus : float
        Transference number of the anion.
    c1 : float
        Concentration on side 1 (mol/L).
    c2 : float
        Concentration on side 2 (mol/L).
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Liquid junction potential in Volts.
    
    """
    c1 = normalize_concentration(c1) if isinstance(c1, str) else c1
    c2 = normalize_concentration(c2) if isinstance(c2, str) else c2
    T = normalize_temperature(T) if isinstance(T, str) else T
    return (gas_constant * T / FARADAY_CONSTANT) * (t_plus - t_minus) * np.log(c2 / c1)


@returns_unit("V")
def henderson_junction_potential_simple(lambda_plus: MolarConductivitySM2Mol,
                                         lambda_minus: MolarConductivitySM2Mol,
                                         c1: ConcentrationMolar, c2: ConcentrationMolar,
                                         T=298.15):
    """
    Compute the liquid junction potential using ionic conductivities.

    E_j = (R*T/F) * (λ+ - λ-)/(λ+ + λ-) * ln(c₁/c₂).

    Parameters
    ----------
    lambda_plus : float
        Limiting molar conductivity of the cation (S·m²/mol).
    lambda_minus : float
        Limiting molar conductivity of the anion (S·m²/mol).
    c1 : float
        Concentration on side 1 (mol/L).
    c2 : float
        Concentration on side 2 (mol/L).
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Liquid junction potential in Volts.
    
    """
    lambda_plus = normalize_molar_conductivity(lambda_plus) if isinstance(lambda_plus, str) else lambda_plus
    lambda_minus = normalize_molar_conductivity(lambda_minus) if isinstance(lambda_minus, str) else lambda_minus
    c1 = normalize_concentration(c1) if isinstance(c1, str) else c1
    c2 = normalize_concentration(c2) if isinstance(c2, str) else c2
    T = normalize_temperature(T) if isinstance(T, str) else T
    return (gas_constant * T / FARADAY_CONSTANT) * \
        (lambda_plus - lambda_minus) / (lambda_plus + lambda_minus) * np.log(c1 / c2)


@returns_unit("mol/L")
def buffer_capacity(C_total: ConcentrationMolar, Ka, H_concentration: ConcentrationMolar):
    """
    Compute the buffer capacity β of a buffer solution.

    β = 2.303 * C_total * Ka * [H⁺] / (Ka + [H⁺])².

    Parameters
    ----------
    C_total : float
        Total buffer concentration in mol/L.
    Ka : float
        Acid dissociation constant.
    H_concentration : float
        Hydrogen ion concentration [H⁺] in mol/L.

    Returns
    -------
    float
        Buffer capacity in mol/L (amount of acid/base needed to change pH by 1).
    
    """
    C_total = normalize_concentration(C_total) if isinstance(C_total, str) else C_total
    H_concentration = normalize_concentration(H_concentration) if isinstance(H_concentration, str) else H_concentration
    return 2.303 * C_total * Ka * H_concentration / (Ka + H_concentration)**2

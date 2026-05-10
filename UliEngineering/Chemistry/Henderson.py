#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Henderson-Hasselbalch equation for pH/buffer calculations.

The Henderson-Hasselbalch equation relates pH, pKa, and the ratio
of conjugate base to weak acid concentrations:

    pH = pKa + log10([A⁻] / [HA])

Also includes the Henderson equation for liquid junction potential:

    E_j = (R*T/F) * Σ((u_i - v_i)/(u_i + v_i)) * ln(a_i'' / a_i')
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np
from scipy.constants import R as gas_constant, physical_constants

__all__ = [
    "henderson_hasselbalch_pH",
    "henderson_hasselbalch_ratio",
    "henderson_hasselbalch_pKa",
    "henderson_junction_potential",
    "henderson_junction_potential_simple",
    "buffer_capacity",
]

FARADAY_CONSTANT = physical_constants["Faraday constant"][0]


@normalize_numeric_args
@returns_unit("")
def henderson_hasselbalch_pH(pKa, base_concentration, acid_concentration):
    """
    Compute pH using the Henderson-Hasselbalch equation.

    pH = pKa + log10([A⁻] / [HA])

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
    return pKa + np.log10(base_concentration / acid_concentration)


@normalize_numeric_args
@returns_unit("")
def henderson_hasselbalch_ratio(pH, pKa):
    """
    Compute the [A⁻]/[HA] ratio from pH and pKa.

    [A⁻] / [HA] = 10^(pH - pKa)

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


@normalize_numeric_args
@returns_unit("")
def henderson_hasselbalch_pKa(pH, base_concentration, acid_concentration):
    """
    Compute pKa from pH and concentrations using the Henderson-Hasselbalch equation.

    pKa = pH - log10([A⁻] / [HA])

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
    return pH - np.log10(base_concentration / acid_concentration)


@normalize_numeric_args
@returns_unit("V")
def henderson_junction_potential(t_plus, t_minus, c1, c2, T=298.15):
    """
    Compute the liquid junction potential using the Henderson equation
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
    return (gas_constant * T / FARADAY_CONSTANT) * (t_plus - t_minus) * np.log(c2 / c1)


@normalize_numeric_args
@returns_unit("V")
def henderson_junction_potential_simple(lambda_plus, lambda_minus, c1, c2, T=298.15):
    """
    Compute the liquid junction potential using ionic conductivities.

    E_j = (R*T/F) * (λ+ - λ-)/(λ+ + λ-) * ln(c₁/c₂)

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
    return (gas_constant * T / FARADAY_CONSTANT) * \
        (lambda_plus - lambda_minus) / (lambda_plus + lambda_minus) * np.log(c1 / c2)


@normalize_numeric_args
@returns_unit("mol/L")
def buffer_capacity(C_total, Ka, H_concentration):
    """
    Compute the buffer capacity β of a buffer solution.

    β = 2.303 * C_total * Ka * [H⁺] / (Ka + [H⁺])²

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
    return 2.303 * C_total * Ka * H_concentration / (Ka + H_concentration)**2

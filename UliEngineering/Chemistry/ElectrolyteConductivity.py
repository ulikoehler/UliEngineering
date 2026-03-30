#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrolyte conductivity calculations.

Provides functions for computing the conductivity (κ) and resistivity (ρ)
of electrolyte solutions from molar conductivity and concentration,
as well as cell-constant-based measurements.

    κ = Λ_m * c     (S/m, when Λ_m in S·m²/mol and c in mol/m³)
    κ = G * K_cell   (S/m, from conductance and cell constant)
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np

__all__ = [
    "electrolyte_conductivity_from_molar",
    "electrolyte_resistivity",
    "conductivity_from_cell_constant",
    "molar_conductivity_from_conductivity",
    "specific_conductance_temperature_correction",
]


@normalize_numeric_args
@returns_unit("S/m")
def electrolyte_conductivity_from_molar(Lambda_m, c):
    """
    Compute electrolyte conductivity from molar conductivity and concentration.

    κ = Λ_m * c * 0.1

    Parameters
    ----------
    Lambda_m : float
        Molar conductivity in S·cm²/mol.
    c : float
        Concentration in mol/L.

    Returns
    -------
    float
        Conductivity in S/m.

    Notes
    -----
    Conversion: S·cm²/mol * mol/L = S·cm²/L = S/(100 cm) = S/m * 0.1
    So κ(S/m) = Λ_m(S·cm²/mol) * c(mol/L) * 0.1
    """
    return Lambda_m * c * 0.1


@normalize_numeric_args
@returns_unit("Ω·m")
def electrolyte_resistivity(conductivity):
    """
    Compute electrolyte resistivity from conductivity.

    ρ = 1 / κ

    Parameters
    ----------
    conductivity : float
        Conductivity in S/m.

    Returns
    -------
    float
        Resistivity in Ω·m.
    """
    return 1.0 / conductivity


@normalize_numeric_args
@returns_unit("S/m")
def conductivity_from_cell_constant(conductance, cell_constant):
    """
    Compute conductivity from measured conductance and cell constant.

    κ = G * K_cell

    Parameters
    ----------
    conductance : float
        Measured conductance in Siemens (S).
    cell_constant : float
        Cell constant in 1/m (= distance / area of electrodes).

    Returns
    -------
    float
        Conductivity in S/m.
    """
    return conductance * cell_constant


@normalize_numeric_args
@returns_unit("S·cm²/mol")
def molar_conductivity_from_conductivity(conductivity, c):
    """
    Compute molar conductivity from specific conductivity and concentration.

    Λ_m = κ / c * 10

    Parameters
    ----------
    conductivity : float
        Specific conductivity in S/m.
    c : float
        Concentration in mol/L.

    Returns
    -------
    float
        Molar conductivity in S·cm²/mol.
    """
    return conductivity / c * 10.0


@normalize_numeric_args
@returns_unit("S/m")
def specific_conductance_temperature_correction(kappa_ref, T, T_ref=298.15, alpha=0.02):
    """
    Temperature-correct specific conductance using a linear model.

    κ(T) = κ_ref * (1 + α * (T - T_ref))

    Parameters
    ----------
    kappa_ref : float
        Reference conductivity in S/m.
    T : float
        Temperature in Kelvin.
    T_ref : float
        Reference temperature in Kelvin (default: 298.15 K).
    alpha : float
        Temperature coefficient per Kelvin (default: 0.02 /K, typical for many electrolytes).

    Returns
    -------
    float
        Temperature-corrected conductivity in S/m.
    """
    return kappa_ref * (1.0 + alpha * (T - T_ref))

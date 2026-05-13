#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrolyte conductivity calculations.

Provides functions for computing the conductivity (κ) and resistivity (ρ).
of electrolyte solutions from molar conductivity and concentration,
as well as cell-constant-based measurements.

    κ = Λ_m * c     (S/m, when Λ_m in S·m²/mol and c in mol/m³)
    κ = G * K_cell   (S/m, from conductance and cell constant)
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature

__all__ = [
    "electrolyte_conductivity_from_molar",
    "electrolyte_resistivity",
    "conductivity_from_cell_constant",
    "molar_conductivity_from_conductivity",
    "specific_conductance_temperature_correction",
    "normalize_molar_conductivity", "MolarConductivitySCm2Mol",
    "normalize_concentration", "ConcentrationMolar",
    "normalize_conductivity", "ConductivitySM",
    "normalize_conductance", "ConductanceS",
    "normalize_cell_constant", "CellConstantPerMeter",
]


def normalize_molar_conductivity(Lambda_m: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(Lambda_m, {"S·cm²/mol": 1.0, "S cm2/mol": 1.0, "S·m²/mol": 10000.0, "S m2/mol": 10000.0}, quantity_name="molar conductivity")

def normalize_concentration(c: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(c, {"mol/L": 1.0, "M": 1.0, "mM": 1e-3, "µM": 1e-6, "mol/m³": 1e-3}, quantity_name="concentration")

def normalize_conductivity(kappa: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(kappa, {"S/m": 1.0, "S/cm": 100.0, "mS/m": 1e-3, "µS/m": 1e-6, "µS/cm": 0.1}, quantity_name="conductivity")

def normalize_conductance(G: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(G, {"S": 1.0, "mS": 1e-3, "µS": 1e-6}, quantity_name="conductance")

def normalize_cell_constant(K_cell: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(K_cell, {"1/m": 1.0, "m⁻¹": 1.0, "1/cm": 100.0, "cm⁻¹": 100.0}, quantity_name="cell constant")

MolarConductivitySCm2Mol = Annotated[NormalizedComputable, normalize_molar_conductivity]
ConcentrationMolar = Annotated[NormalizedComputable, normalize_concentration]
ConductivitySM = Annotated[NormalizedComputable, normalize_conductivity]
ConductanceS = Annotated[NormalizedComputable, normalize_conductance]
CellConstantPerMeter = Annotated[NormalizedComputable, normalize_cell_constant]


@returns_unit("S/m")
def electrolyte_conductivity_from_molar(Lambda_m: MolarConductivitySCm2Mol, c: ConcentrationMolar):
    """
    Compute electrolyte conductivity from molar conductivity and concentration.

    κ = Λ_m * c * 0.1.

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
    Lambda_m = normalize_molar_conductivity(Lambda_m) if isinstance(Lambda_m, str) else Lambda_m
    c = normalize_concentration(c) if isinstance(c, str) else c
    return Lambda_m * c * 0.1


@returns_unit("Ω·m")
def electrolyte_resistivity(conductivity: ConductivitySM):
    """
    Compute electrolyte resistivity from conductivity.

    ρ = 1 / κ.

    Parameters
    ----------
    conductivity : float
        Conductivity in S/m.

    Returns
    -------
    float
        Resistivity in Ω·m.
    
    """
    conductivity = normalize_conductivity(conductivity) if isinstance(conductivity, str) else conductivity
    return 1.0 / conductivity


@returns_unit("S/m")
def conductivity_from_cell_constant(conductance: ConductanceS, cell_constant: CellConstantPerMeter):
    """
    Compute conductivity from measured conductance and cell constant.

    κ = G * K_cell.

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
    conductance = normalize_conductance(conductance) if isinstance(conductance, str) else conductance
    cell_constant = normalize_cell_constant(cell_constant) if isinstance(cell_constant, str) else cell_constant
    return conductance * cell_constant


@returns_unit("S·cm²/mol")
def molar_conductivity_from_conductivity(conductivity: ConductivitySM, c: ConcentrationMolar):
    """
    Compute molar conductivity from specific conductivity and concentration.

    Λ_m = κ / c * 10.

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
    conductivity = normalize_conductivity(conductivity) if isinstance(conductivity, str) else conductivity
    c = normalize_concentration(c) if isinstance(c, str) else c
    return conductivity / c * 10.0


@returns_unit("S/m")
def specific_conductance_temperature_correction(kappa_ref: ConductivitySM, T, T_ref=298.15, alpha=0.02):
    """
    Temperature-correct specific conductance using a linear model.

    κ(T) = κ_ref * (1 + α * (T - T_ref)).

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
    kappa_ref = normalize_conductivity(kappa_ref) if isinstance(kappa_ref, str) else kappa_ref
    T = normalize_temperature(T) if isinstance(T, str) else T
    T_ref = normalize_temperature(T_ref) if isinstance(T_ref, str) else T_ref
    return kappa_ref * (1.0 + alpha * (T - T_ref))

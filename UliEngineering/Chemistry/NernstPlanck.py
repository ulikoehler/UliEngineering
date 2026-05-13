#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nernst-Planck equation for ionic flux.

The Nernst-Planck equation describes transport of ionic species.
under the influence of both concentration gradients and electric fields:

    J = -D * (dC/dx + z*F*C/(R*T) * dφ/dx)

where:
    J  = ionic flux (mol/(m²·s))
    D  = diffusion coefficient (m²/s)
    C  = concentration (mol/m³)
    dC/dx = concentration gradient (mol/m⁴)
    z  = charge number of the ion
    F  = Faraday constant (C/mol)
    R  = gas constant (J/(mol·K))
    T  = temperature (K)
    dφ/dx = electric potential gradient (V/m)
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature
import numpy as np
from scipy.constants import R as gas_constant, physical_constants

__all__ = [
    "nernst_planck_flux",
    "nernst_planck_diffusion_flux",
    "nernst_planck_migration_flux",
    "einstein_relation_diffusion_mobility",
    "ionic_mobility_from_diffusion",
    "normalize_diffusion_coefficient", "DiffusionCoefficientM2S",
    "normalize_concentration", "ConcentrationMolM3",
    "normalize_ionic_mobility", "IonicMobilityM2VS",
]

FARADAY_CONSTANT = physical_constants["Faraday constant"][0]


def normalize_diffusion_coefficient(D: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(D, {"m²/s": 1.0, "m2/s": 1.0, "cm²/s": 1e-4, "cm2/s": 1e-4, "mm²/s": 1e-6, "mm2/s": 1e-6}, quantity_name="diffusion coefficient")

def normalize_concentration(c: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(c, {"mol/m³": 1.0, "mol/m3": 1.0, "mol/m^3": 1.0, "mol/L": 1000.0, "M": 1000.0, "mM": 1.0, "µM": 1e-3}, quantity_name="concentration")

def normalize_ionic_mobility(mobility: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(mobility, {"m²/(V·s)": 1.0, "m2/(V·s)": 1.0, "cm²/(V·s)": 1e-4, "cm2/(V·s)": 1e-4}, quantity_name="ionic mobility")

DiffusionCoefficientM2S = Annotated[NormalizedComputable, normalize_diffusion_coefficient]
ConcentrationMolM3 = Annotated[NormalizedComputable, normalize_concentration]
IonicMobilityM2VS = Annotated[NormalizedComputable, normalize_ionic_mobility]


@returns_unit("mol/(m²·s)")
def nernst_planck_flux(D: DiffusionCoefficientM2S, dC_dx, z, C: ConcentrationMolM3, dPhi_dx, T=298.15):
    """
    Compute the ionic flux using the Nernst-Planck equation.

    J = -D * (dC/dx + z*F*C/(R*T) * dφ/dx).

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    dC_dx : float
        Concentration gradient in mol/m⁴.
    z : float
        Charge number of the ion.
    C : float
        Local concentration in mol/m³.
    dPhi_dx : float
        Electric potential gradient in V/m.
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Ionic flux in mol/(m²·s).

    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    C = normalize_concentration(C) if isinstance(C, str) else C
    T = normalize_temperature(T) if isinstance(T, str) else T
    return -D * (dC_dx + z * FARADAY_CONSTANT * C / (gas_constant * T) * dPhi_dx)


@returns_unit("mol/(m²·s)")
def nernst_planck_diffusion_flux(D: DiffusionCoefficientM2S, dC_dx):
    """
    Compute the diffusion component of Nernst-Planck flux (Fick's first law).

    J_diff = -D * dC/dx.

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    dC_dx : float
        Concentration gradient in mol/m⁴.

    Returns
    -------
    float
        Diffusion flux in mol/(m²·s).
    
    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    return -D * dC_dx


@returns_unit("mol/(m²·s)")
def nernst_planck_migration_flux(D: DiffusionCoefficientM2S, z, C: ConcentrationMolM3, dPhi_dx, T=298.15):
    """
    Compute the migration (electromigration) component of Nernst-Planck flux.

    J_mig = -D * z * F * C / (R * T) * dφ/dx.

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    z : float
        Charge number of the ion.
    C : float
        Local concentration in mol/m³.
    dPhi_dx : float
        Electric potential gradient in V/m.
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Migration flux in mol/(m²·s).
    
    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    C = normalize_concentration(C) if isinstance(C, str) else C
    T = normalize_temperature(T) if isinstance(T, str) else T
    return -D * z * FARADAY_CONSTANT * C / (gas_constant * T) * dPhi_dx


@returns_unit("m²/s")
def einstein_relation_diffusion_mobility(mobility: IonicMobilityM2VS, T=298.15):
    """
    Compute diffusion coefficient from ionic mobility using the Einstein relation.

    D = μ * k_B * T / e = μ * R * T / F.

    (For ions, using molar quantities: D = u * R * T / (|z| * F)
     where u is the electrochemical mobility)

    Parameters
    ----------
    mobility : float
        Ionic electrical mobility μ in m²/(V·s).
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Diffusion coefficient in m²/s.
    
    """
    from scipy.constants import k as k_B, e
    mobility = normalize_ionic_mobility(mobility) if isinstance(mobility, str) else mobility
    T = normalize_temperature(T) if isinstance(T, str) else T
    return mobility * k_B * T / e


@returns_unit("m²/(V·s)")
def ionic_mobility_from_diffusion(D: DiffusionCoefficientM2S, z, T=298.15):
    """
    Compute ionic mobility from diffusion coefficient.

    μ = D * |z| * F / (R * T).

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    z : float
        Charge number of the ion.
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Ionic mobility in m²/(V·s).
    
    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    T = normalize_temperature(T) if isinstance(T, str) else T
    return D * np.abs(z) * FARADAY_CONSTANT / (gas_constant * T)

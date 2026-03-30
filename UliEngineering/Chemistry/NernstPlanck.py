#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nernst-Planck equation for ionic flux.

The Nernst-Planck equation describes transport of ionic species
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
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np
from scipy.constants import R as gas_constant, physical_constants

__all__ = [
    "nernst_planck_flux",
    "nernst_planck_diffusion_flux",
    "nernst_planck_migration_flux",
    "einstein_relation_diffusion_mobility",
    "ionic_mobility_from_diffusion",
]

FARADAY_CONSTANT = physical_constants["Faraday constant"][0]


@normalize_numeric_args
@returns_unit("mol/(m²·s)")
def nernst_planck_flux(D, dC_dx, z, C, dPhi_dx, T=298.15):
    """
    Compute the ionic flux using the Nernst-Planck equation.

    J = -D * (dC/dx + z*F*C/(R*T) * dφ/dx)

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
    return -D * (dC_dx + z * FARADAY_CONSTANT * C / (gas_constant * T) * dPhi_dx)


@normalize_numeric_args
@returns_unit("mol/(m²·s)")
def nernst_planck_diffusion_flux(D, dC_dx):
    """
    Compute the diffusion component of Nernst-Planck flux (Fick's first law).

    J_diff = -D * dC/dx

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
    return -D * dC_dx


@normalize_numeric_args
@returns_unit("mol/(m²·s)")
def nernst_planck_migration_flux(D, z, C, dPhi_dx, T=298.15):
    """
    Compute the migration (electromigration) component of Nernst-Planck flux.

    J_mig = -D * z * F * C / (R * T) * dφ/dx

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
    return -D * z * FARADAY_CONSTANT * C / (gas_constant * T) * dPhi_dx


@normalize_numeric_args
@returns_unit("m²/s")
def einstein_relation_diffusion_mobility(mobility, T=298.15):
    """
    Compute diffusion coefficient from ionic mobility using the Einstein relation.

    D = μ * k_B * T / e = μ * R * T / F

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
    return mobility * k_B * T / e


@normalize_numeric_args
@returns_unit("m²/(V·s)")
def ionic_mobility_from_diffusion(D, z, T=298.15):
    """
    Compute ionic mobility from diffusion coefficient.

    μ = D * |z| * F / (R * T)

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
    return D * np.abs(z) * FARADAY_CONSTANT / (gas_constant * T)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debye-Hückel theory for activity coefficients.

The Debye-Hückel limiting law:
    log10(γ±) = -A * |z+ * z-| * √I

The extended Debye-Hückel equation:
    log10(γ) = -A * z² * √I / (1 + B * a * √I)

where:
    A ≈ 0.509 (mol/L)^(-1/2) at 25 °C in water
    B ≈ 3.281 (nm⁻¹ (mol/L)^(-1/2)) at 25 °C in water
    a = effective ion diameter (nm)
    I = ionic strength (mol/L)
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np

__all__ = [
    "debye_huckel_limiting_law",
    "debye_huckel_extended",
    "debye_huckel_activity_coefficient",
    "debye_huckel_extended_activity_coefficient",
    "debye_length",
]


@normalize_numeric_args
@returns_unit("")
def debye_huckel_limiting_law(z_plus, z_minus, I, A=0.509):
    """
    Compute log10 of the mean activity coefficient using the
    Debye-Hückel limiting law. Valid for very dilute solutions (I < 0.01 M).

    log10(γ±) = -A * |z+ * z-| * √I

    Parameters
    ----------
    z_plus : float
        Charge number of the cation.
    z_minus : float
        Charge number of the anion (positive value, e.g. 1 for Cl⁻).
    I : float
        Ionic strength in mol/L.
    A : float
        Debye-Hückel A parameter (default: 0.509).

    Returns
    -------
    float
        log10 of the mean activity coefficient.
    """
    return -A * np.abs(z_plus * z_minus) * np.sqrt(I)


@normalize_numeric_args
@returns_unit("")
def debye_huckel_extended(z, I, a=0.3, A=0.509, B=3.281):
    """
    Compute log10 of the activity coefficient using the
    extended Debye-Hückel equation.

    log10(γ) = -A * z² * √I / (1 + B * a * √I)

    Parameters
    ----------
    z : float
        Charge number of the ion.
    I : float
        Ionic strength in mol/L.
    a : float
        Effective ion diameter in nm (default: 0.3 nm).
    A : float
        Debye-Hückel A parameter (default: 0.509).
    B : float
        Debye-Hückel B parameter in nm⁻¹·(mol/L)^(-1/2) (default: 3.281).

    Returns
    -------
    float
        log10 of the activity coefficient.
    """
    sqrt_I = np.sqrt(I)
    return -A * z**2 * sqrt_I / (1.0 + B * a * sqrt_I)


@normalize_numeric_args
@returns_unit("")
def debye_huckel_activity_coefficient(z_plus, z_minus, I, A=0.509):
    """
    Compute the mean activity coefficient using the Debye-Hückel limiting law.

    γ± = 10^(-A * |z+ * z-| * √I)

    Parameters
    ----------
    z_plus : float
        Charge number of the cation.
    z_minus : float
        Charge number of the anion (positive value).
    I : float
        Ionic strength in mol/L.
    A : float
        Debye-Hückel A parameter (default: 0.509).

    Returns
    -------
    float
        Mean activity coefficient (dimensionless).
    """
    return 10.0 ** debye_huckel_limiting_law(z_plus, z_minus, I, A)


@normalize_numeric_args
@returns_unit("")
def debye_huckel_extended_activity_coefficient(z, I, a=0.3, A=0.509, B=3.281):
    """
    Compute the activity coefficient using the extended Debye-Hückel equation.

    γ = 10^(log10_γ_extended)

    Parameters
    ----------
    z : float
        Charge number of the ion.
    I : float
        Ionic strength in mol/L.
    a : float
        Effective ion diameter in nm.
    A : float
        Debye-Hückel A parameter (default: 0.509).
    B : float
        Debye-Hückel B parameter (default: 3.281).

    Returns
    -------
    float
        Activity coefficient (dimensionless).
    """
    return 10.0 ** debye_huckel_extended(z, I, a, A, B)


@normalize_numeric_args
@returns_unit("m")
def debye_length(I, T=298.15, epsilon_r=78.4):
    """
    Compute the Debye length (screening length) for an electrolyte solution.

    λ_D = sqrt(ε₀ * εᵣ * k_B * T / (2 * N_A * e² * I))

    For water at 25 °C: λ_D ≈ 0.304 / √I nm (with I in mol/L)

    Parameters
    ----------
    I : float
        Ionic strength in mol/L.
    T : float
        Temperature in Kelvin (default: 298.15).
    epsilon_r : float
        Relative permittivity (default: 78.4 for water at 25 °C).

    Returns
    -------
    float
        Debye length in meters.
    """
    from scipy.constants import epsilon_0, k as k_B, N_A, e
    # Convert I from mol/L to mol/m³
    I_m3 = I * 1000.0
    return np.sqrt(epsilon_0 * epsilon_r * k_B * T / (2.0 * N_A * e**2 * I_m3))

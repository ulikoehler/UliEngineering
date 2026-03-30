#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fick's laws of diffusion.

Fick's first law (steady-state):
    J = -D * dC/dx

Fick's second law (time-dependent):
    dC/dt = D * d²C/dx²

Also includes diffusion-related utility functions like
mean diffusion distance, diffusion time estimation, and
the error-function concentration profile solution.
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np
from scipy.special import erfc

__all__ = [
    "fick_first_law",
    "fick_diffusion_distance",
    "fick_diffusion_time",
    "fick_semi_infinite_concentration",
    "fick_thin_film_concentration",
    "diffusion_coefficient_from_temperature",
]


@normalize_numeric_args
@returns_unit("mol/(m²·s)")
def fick_first_law(D, dC_dx):
    """
    Compute the diffusion flux using Fick's first law.

    J = -D * dC/dx

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
@returns_unit("m")
def fick_diffusion_distance(D, t):
    """
    Compute the characteristic (RMS) diffusion distance.

    x_rms = √(2 * D * t)

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    t : float
        Time in seconds.

    Returns
    -------
    float
        RMS diffusion distance in meters.
    """
    return np.sqrt(2.0 * D * t)


@normalize_numeric_args
@returns_unit("s")
def fick_diffusion_time(D, x):
    """
    Compute the time required for diffusion over a distance x.

    t = x² / (2 * D)

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    x : float
        Distance in meters.

    Returns
    -------
    float
        Time in seconds.
    """
    return x**2 / (2.0 * D)


@normalize_numeric_args
@returns_unit("mol/m³")
def fick_semi_infinite_concentration(C0, Cs, x, D, t):
    """
    Compute concentration at distance x and time t using the semi-infinite
    solid solution of Fick's second law (constant surface concentration).

    C(x,t) = Cs - (Cs - C0) * erfc(x / (2*√(D*t)))

    Alternatively: C(x,t) = C0 + (Cs - C0) * erfc(x / (2*√(D*t)))

    Parameters
    ----------
    C0 : float
        Initial uniform concentration in mol/m³.
    Cs : float
        Surface concentration in mol/m³ (constant boundary).
    x : float
        Distance from surface in meters.
    D : float
        Diffusion coefficient in m²/s.
    t : float
        Time in seconds.

    Returns
    -------
    float
        Concentration at position x and time t in mol/m³.
    """
    return C0 + (Cs - C0) * erfc(x / (2.0 * np.sqrt(D * t)))


@normalize_numeric_args
@returns_unit("mol/m³")
def fick_thin_film_concentration(M, D, t, x):
    """
    Concentration profile from a thin-film (impulse) source diffusing
    in one dimension (Fick's second law, instantaneous plane source).

    C(x,t) = M / √(4πDt) * exp(-x² / (4Dt))

    Parameters
    ----------
    M : float
        Amount of substance per unit area initially deposited (mol/m²).
    D : float
        Diffusion coefficient in m²/s.
    t : float
        Time in seconds.
    x : float
        Distance from the source plane in meters.

    Returns
    -------
    float
        Concentration in mol/m³.
    """
    return M / np.sqrt(4.0 * np.pi * D * t) * np.exp(-x**2 / (4.0 * D * t))


@normalize_numeric_args
@returns_unit("m²/s")
def diffusion_coefficient_from_temperature(D0, Ea, T):
    """
    Arrhenius-type temperature dependence of diffusion coefficient.

    D(T) = D0 * exp(-Ea / (R * T))

    Parameters
    ----------
    D0 : float
        Pre-exponential factor in m²/s.
    Ea : float
        Activation energy in J/mol.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Diffusion coefficient at temperature T in m²/s.
    """
    from scipy.constants import R
    return D0 * np.exp(-Ea / (R * T))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Davies equation for activity coefficient estimation.

The Davies equation is an empirical extension of Debye-Hückel theory
for estimating activity coefficients at higher ionic strengths (up to ~0.5 M):

    log10(γ) = -A * z² * (√I / (1 + √I) - 0.3 * I)

where:
    γ = activity coefficient
    A = Debye-Hückel parameter (~0.509 at 25 °C in water)
    z = charge number of the ion
    I = ionic strength (mol/L)
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np

__all__ = [
    "davies_activity_coefficient",
    "davies_log_activity_coefficient",
    "debye_huckel_A_parameter",
]


def debye_huckel_A_parameter(T=298.15, epsilon_r=78.4):
    """
    Compute the Debye-Hückel A parameter.

    A = 1.8246e6 / (epsilon_r * T)^(3/2)

    At 25 °C in water: A ≈ 0.509

    Parameters
    ----------
    T : float
        Temperature in Kelvin (default: 298.15 K).
    epsilon_r : float
        Relative permittivity of the solvent (default: 78.4 for water at 25 °C).

    Returns
    -------
    float
        Debye-Hückel A parameter in (mol/L)^(-1/2).
    """
    return 1.8246e6 / (epsilon_r * T)**1.5


@normalize_numeric_args
@returns_unit("")
def davies_log_activity_coefficient(z, I, A=0.509):
    """
    Compute log10 of the activity coefficient using the Davies equation.

    log10(γ) = -A * z² * (√I / (1 + √I) - 0.3 * I)

    Parameters
    ----------
    z : float
        Charge number of the ion.
    I : float
        Ionic strength in mol/L.
    A : float
        Debye-Hückel A parameter (default: 0.509 for water at 25 °C).

    Returns
    -------
    float
        log10 of the activity coefficient (dimensionless).
    """
    sqrt_I = np.sqrt(I)
    return -A * z**2 * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I)


@normalize_numeric_args
@returns_unit("")
def davies_activity_coefficient(z, I, A=0.509):
    """
    Compute the activity coefficient using the Davies equation.

    γ = 10^(-A * z² * (√I / (1 + √I) - 0.3 * I))

    Parameters
    ----------
    z : float
        Charge number of the ion.
    I : float
        Ionic strength in mol/L.
    A : float
        Debye-Hückel A parameter (default: 0.509 for water at 25 °C).

    Returns
    -------
    float
        Activity coefficient (dimensionless).
    """
    return 10.0 ** davies_log_activity_coefficient(z, I, A)

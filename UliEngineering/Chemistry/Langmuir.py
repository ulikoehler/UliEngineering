#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langmuir adsorption isotherm.

The Langmuir isotherm describes adsorption of molecules onto a surface
with a finite number of identical sites:

    θ = K * C / (1 + K * C)

    q = q_max * K * C / (1 + K * C)

where:
    θ    = fractional surface coverage (0 to 1)
    K    = Langmuir adsorption constant (L/mol or 1/pressure)
    C    = concentration or partial pressure
    q    = amount adsorbed
    q_max = maximum capacity
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np

__all__ = [
    "langmuir_coverage",
    "langmuir_adsorbed_amount",
    "langmuir_constant_from_coverage",
    "langmuir_competitive_coverage",
    "langmuir_inverse_linearized",
    "langmuir_dissociation_rate",
]


@normalize_numeric_args
@returns_unit("")
def langmuir_coverage(K, C):
    """
    Compute fractional surface coverage using the Langmuir isotherm.

    θ = K * C / (1 + K * C)

    Parameters
    ----------
    K : float
        Langmuir adsorption constant (L/mol or 1/Pa for gas).
    C : float
        Concentration (mol/L) or partial pressure (Pa).

    Returns
    -------
    float
        Fractional coverage θ (dimensionless, 0 to 1).
    """
    return K * C / (1.0 + K * C)


@normalize_numeric_args
@returns_unit("mol/g")
def langmuir_adsorbed_amount(q_max, K, C):
    """
    Compute amount adsorbed using the Langmuir isotherm.

    q = q_max * K * C / (1 + K * C)

    Parameters
    ----------
    q_max : float
        Maximum adsorption capacity in mol/g.
    K : float
        Langmuir adsorption constant (L/mol).
    C : float
        Equilibrium concentration in mol/L.

    Returns
    -------
    float
        Amount adsorbed in mol/g.
    """
    return q_max * K * C / (1.0 + K * C)


@normalize_numeric_args
@returns_unit("L/mol")
def langmuir_constant_from_coverage(theta, C):
    """
    Compute the Langmuir constant K from measured coverage and concentration.

    K = θ / (C * (1 - θ))

    Parameters
    ----------
    theta : float
        Fractional surface coverage (0 to 1).
    C : float
        Concentration in mol/L.

    Returns
    -------
    float
        Langmuir constant K in L/mol.
    """
    return theta / (C * (1.0 - theta))


@returns_unit("")
def langmuir_competitive_coverage(K_i, C_i, K_all, C_all):
    """
    Compute fractional coverage of species i in competitive Langmuir adsorption.

    θ_i = K_i * C_i / (1 + Σ(K_j * C_j))

    Parameters
    ----------
    K_i : float
        Langmuir constant for species i.
    C_i : float
        Concentration of species i.
    K_all : array-like
        Langmuir constants for all competing species.
    C_all : array-like
        Concentrations of all competing species.

    Returns
    -------
    float
        Fractional coverage of species i.
    """
    K_arr = np.asarray(K_all, dtype=float)
    C_arr = np.asarray(C_all, dtype=float)
    denominator = 1.0 + np.sum(K_arr * C_arr)
    return float(K_i) * float(C_i) / denominator


@normalize_numeric_args
@returns_unit("g/mol")
def langmuir_inverse_linearized(C, q):
    """
    Compute 1/q vs 1/C for Langmuir linearized form (double-reciprocal / Lineweaver-Burk).

    1/q = 1/q_max + 1/(q_max * K) * 1/C

    This returns (1/C, 1/q) which can be used for linear regression
    to find q_max and K.

    Parameters
    ----------
    C : float
        Equilibrium concentration.
    q : float
        Amount adsorbed.

    Returns
    -------
    tuple
        (1/C, 1/q) for plotting and linear regression.
    """
    return (1.0 / C, 1.0 / q)


@normalize_numeric_args
@returns_unit("1/s")
def langmuir_dissociation_rate(k_ads, K):
    """
    Compute the dissociation rate constant from adsorption rate constant and equilibrium constant.

    k_des = k_ads / K

    Parameters
    ----------
    k_ads : float
        Adsorption rate constant in L/(mol·s) or similar.
    K : float
        Langmuir equilibrium constant.

    Returns
    -------
    float
        Dissociation rate constant in 1/s.
    """
    return k_ads / K

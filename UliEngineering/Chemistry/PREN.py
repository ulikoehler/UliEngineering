#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREN (Pitting Resistance Equivalent Number) for corrosion resistance.

The PREN is an empirical index used to rank stainless steels and
nickel alloys by their resistance to pitting corrosion:

    PREN = %Cr + 3.3 * %Mo + 16 * %N

Variants:
    PRE_N  = %Cr + 3.3 * %Mo + 16 * %N     (standard)
    PRE_NW = %Cr + 3.3 * (%Mo + 0.5 * %W) + 16 * %N  (includes tungsten)
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "pren",
    "pren_w",
    "COMMON_STEEL_COMPOSITIONS",
]

# Common stainless steel compositions (weight %)
# Source: ASTM / EN standards
COMMON_STEEL_COMPOSITIONS = {
    "304": {"Cr": 18.0, "Mo": 0.0, "N": 0.05, "W": 0.0},
    "316": {"Cr": 16.5, "Mo": 2.1, "N": 0.05, "W": 0.0},
    "316L": {"Cr": 16.5, "Mo": 2.1, "N": 0.05, "W": 0.0},
    "2205": {"Cr": 22.0, "Mo": 3.1, "N": 0.17, "W": 0.0},
    "2507": {"Cr": 25.0, "Mo": 3.8, "N": 0.27, "W": 0.0},
    "254SMO": {"Cr": 20.0, "Mo": 6.1, "N": 0.20, "W": 0.0},
    "904L": {"Cr": 20.0, "Mo": 4.3, "N": 0.05, "W": 0.0},
    "Alloy625": {"Cr": 21.5, "Mo": 9.0, "N": 0.0, "W": 0.0},
    "AlloyC276": {"Cr": 15.5, "Mo": 16.0, "N": 0.0, "W": 3.75},
}


@normalize_numeric_args
@returns_unit("")
def pren(Cr, Mo, N):
    """
    Compute the Pitting Resistance Equivalent Number (PREN).

    PREN = %Cr + 3.3 * %Mo + 16 * %N

    Parameters
    ----------
    Cr : float
        Chromium content in weight %.
    Mo : float
        Molybdenum content in weight %.
    N : float
        Nitrogen content in weight %.

    Returns
    -------
    float
        PREN value (dimensionless). Higher values indicate better pitting resistance.
        Values > 32 are considered seawater-resistant.

    """
    return Cr + 3.3 * Mo + 16.0 * N


@normalize_numeric_args
@returns_unit("")
def pren_w(Cr, Mo, N, W):
    """
    Compute the Pitting Resistance Equivalent Number including tungsten (PRE_NW).

    PRE_NW = %Cr + 3.3 * (%Mo + 0.5 * %W) + 16 * %N

    Parameters
    ----------
    Cr : float
        Chromium content in weight %.
    Mo : float
        Molybdenum content in weight %.
    N : float
        Nitrogen content in weight %.
    W : float
        Tungsten content in weight %.

    Returns
    -------
    float
        PRE_NW value (dimensionless).

    """
    return Cr + 3.3 * (Mo + 0.5 * W) + 16.0 * N

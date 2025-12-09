#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate idealized reactances.

Originally published at techoverflow.net
"""
import numpy as np

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "capacitive_reactance",
    "inductive_reactance",
    "inductance_from_reactance",
    "capacitance_from_reactance",
]

@normalize_numeric_args
@returns_unit("Ω")
def capacitive_reactance(c, f=1000.0):
    """
    Compute the capacitive reactance for a given capacitance and frequency.
    """
    return 1.0 / (2 * np.pi * f * c)


@normalize_numeric_args
@returns_unit("Ω")
def inductive_reactance(l, f=1000.0):
    """
    Compute the inductive reactance for a given inductance and frequency.
    """
    return 2 * np.pi * f * l

@normalize_numeric_args
@returns_unit("H")
def inductance_from_reactance(x, f=1000.0):
    """
    Compute the inductance (H) from an inductive reactance (Ω) at a given
    frequency f (Hz).

    Formula: X_L = 2 * pi * f * L  =>  L = X_L / (2 * pi * f)
    """
    return x / (2 * np.pi * f)

@normalize_numeric_args
@returns_unit("F")
def capacitance_from_reactance(x, f=1000.0):
    """
    Compute the capacitance (F) from a capacitive reactance (Ω) at a given
    frequency f (Hz).

    Formula: X_C = 1 / (2 * pi * f * C)  =>  C = 1 / (2 * pi * f * X_C)
    """
    return 1.0 / (2 * np.pi * f * x)

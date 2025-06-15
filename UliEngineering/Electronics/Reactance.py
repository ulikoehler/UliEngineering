#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate idealized reactances.

Originally published at techoverflow.net
"""
import numpy as np

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = ["capacitive_reactance", "inductive_reactance"]

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

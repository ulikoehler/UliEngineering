#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystal oscillator utilities
"""
from UliEngineering.EngineerIO import normalize_numeric, Quantity
import numpy as np

__all__ = ["load_capacitors"]

def load_capacitors(cload, cstray="2 pF") -> Quantity("F"):
    """
    Compute the load capacitors which should be used for a given crystal,
    given that the load capacitors should be symmetric (i.e. have the same value).

    Parameters
    ----------
    cload : float
        The load capacitance as given in the crystal datasheet
    cstray : float
        The stray capacitance
    """
    cload = normalize_numeric(cload)
    cstray = normalize_numeric(cstray)
    # cload = (C1 * C2) / (C1 + C2) + Cstray where C1 == C2
    # => solve A = (B*B) / (B+B) + C for B
    return 2 * (cload - cstray)

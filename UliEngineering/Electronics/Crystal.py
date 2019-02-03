#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystal oscillator utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit

__all__ = ["load_capacitors"]

def load_capacitors(cload, cpin="3pF", cstray="1 pF") -> Unit("F"):
    """
    Compute the load capacitors which should be used for a given crystal,
    given that the load capacitors should be symmetric (i.e. have the same value).

    Based on (C1 * C2) / (C1 + C2) + Cstray
    for C1 == C2 == (returned value) + cpin

    >>> auto_format(load_capacitors, "6 pF", cpin="3 pF", cstray="1pF")

    Parameters
    ----------
    cload : float
        The load capacitance as given in the crystal datasheet
    cstray : float
        The stray capacitance
    cpin : float
        The capacitance of one of the oscillator pins of the connected device
    """
    cload = normalize_numeric(cload)
    cstray = normalize_numeric(cstray)
    cpin = normalize_numeric(cpin)
    # cload = (C1 * C2) / (C1 + C2) + Cstray where C1 == C2
    # => solve A = (B*B) / (B+B) + C for B
    # => solve A = ((B+P)*(B+P)) / ((B+P)+(B+P)) + C for B
    return (2 * (cload - cstray)) - cpin

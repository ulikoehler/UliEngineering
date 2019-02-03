#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystal oscillator utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit

__all__ = ["load_capacitors", "actual_load_capacitance"]

def load_capacitors(cload, cpin="3 pF", cstray="2 pF") -> Unit("F"):
    """
    Compute the load capacitors which should be used for a given crystal,
    given that the load capacitors should be symmetric (i.e. have the same value).

    NOTE: You need to use a stray capacitance value that does NOT
    include the parasitic pin capacitance!

    Based on (C1 * C2) / (C1 + C2) + Cstray
    for C1 == C2 == (returned value) + cpin

    >>> auto_format(load_capacitors, "6 pF", cpin="3 pF", cstray="2pF")
    '5.00 pF'

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

def actual_load_capacitance(cext, cpin="3 pF", cstray="2 pF") -> Unit("F"):
    """
    Compute the actual load capacitance of a crystal given:

    - The external capacitance value (use "10 pF" if your have a
    10 pF capacitor on each of the crystal pins)
    - The parasitic pin capacitance

    The value returned should match the load capacitance value
    in the crystal datasheet.

    Based on (C1 * C2) / (C1 + C2) + Cstray

    If yu use a

    >>> auto_format(actual_load_capacitance, "5 pF", cpin="3 pF", cstray="2pF")
    '6.00 pF'

    Parameters
    ----------
    cext : float
        The load capacitor value
    cstray : float
        The stray capacitance
    cpin : float
        The capacitance of one of the oscillator pins of the connected device
    """
    cext = normalize_numeric(cext)
    cstray = normalize_numeric(cstray)
    cpin = normalize_numeric(cpin)
    # cload = (C1 * C2) / (C1 + C2) + Cstray where C1 == C2
    # => solve A = (B*B) / (B+B) + C for B
    # => solve A = ((B+P)*(B+P)) / ((B+P)+(B+P)) + C for B
    ctotal = cext + cpin
    return cstray + ((ctotal * ctotal) / (ctotal + ctotal))

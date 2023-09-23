#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystal oscillator utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = [
    "lc_cutoff_frequency", "rc_cutoff_frequency"
]

def lc_cutoff_frequency(l, c) -> Unit("Hz"):
    """
    Compute the resonance frequency of an LC oscillator circuit
    given the inductance and capacitance.
    
    This function can likewise be used to compute the corner frequency f_p
    of a LC filter.

    Parameters
    ----------
    l : float
        The inductance in Henry
    c : float
        The capacitance in Farad
    """
    l = normalize_numeric(l)
    c = normalize_numeric(c)
    return 1. / (2 * np.pi * np.sqrt(l * c))

def rc_cutoff_frequency(r, c) -> Unit("Hz"):
    """
    Compute the corner frequency of an RC filter given the resistance
    and capacitance.

    Parameters
    ----------
    r : float
        The resistance in Ohm
    c : float
        The capacitance in Farad
    """
    r = normalize_numeric(r)
    c = normalize_numeric(c)
    return 1. / (2 * np.pi * r * c)
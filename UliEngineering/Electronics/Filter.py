#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystal oscillator utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
from collections import namedtuple
import numpy as np

__all__ = [
    "lc_cutoff_frequency", "rc_cutoff_frequency",
    "rc_feedforward_pole_and_zero", "PoleAndZero"
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

PoleAndZero = namedtuple("PoleAndZero", ["pole", "zero"])

def rc_feedforward_pole_and_zero(r1, r2, cff) -> (Unit("Hz"), Unit("Hz")):
    """
    Compute the pole and zero of a resistor divider with a feedforward capacitor.
    This is useful to compute the compensation capacitor.
    
    Cff is assumed to be in parallel with R1, while R2 goes to ground.
    
    For reference, see
    https://www.ti.com/lit/an/slva289b/slva289b.pdf
    equations 1 and 2.

    Parameters
    ----------
    r1 : float
        The resistance of the feedback path in Ohm
    r2 : float
        The resistance of the feedforward path in Ohm
    cff : float
        The capacitance of the feedforward path in Farad
    """
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    cff = normalize_numeric(cff)
    return PoleAndZero(
        zero=rc_cutoff_frequency(r1, cff),
        pole=(1./r2 + 1./r2)/(2*np.pi*cff)
    )
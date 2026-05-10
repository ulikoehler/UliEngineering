#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility to calculate idealized reactances.

Originally published at techoverflow.net.
"""
import numpy as np

from UliEngineering.EngineerIO.Decorators import returns_unit
from .Capacitors import normalize_capacitance, CapacitanceFarad
from .Filter import normalize_inductance, InductanceH, normalize_frequency, FrequencyHz
from .Diode import normalize_resistance, ResistanceOhm

__all__ = [
    "capacitive_reactance",
    "inductive_reactance",
    "inductance_from_reactance",
    "capacitance_from_reactance",
]

@returns_unit("Ω")
def capacitive_reactance(c: CapacitanceFarad, f: FrequencyHz = 1000.0):
    """
    Compute the capacitive reactance for a given capacitance and frequency.

    Parameters.
    ----------
    c : CapacitanceFarad
        Capacitance in Farads.
    f : FrequencyHz, optional
        Frequency in Hz. Default is 1000.0.

    Returns
    -------
    float
        Capacitive reactance in Ohms.
    
    """
    c = normalize_capacitance(c) if isinstance(c, str) else c
    f = normalize_frequency(f) if isinstance(f, str) else f
    return 1.0 / (2 * np.pi * f * c)


@returns_unit("Ω")
def inductive_reactance(l: InductanceH, f: FrequencyHz = 1000.0):
    """
    Compute the inductive reactance for a given inductance and frequency.

    Parameters.
    ----------
    l : InductanceH
        Inductance in Henrys.
    f : FrequencyHz, optional
        Frequency in Hz. Default is 1000.0.

    Returns
    -------
    float
        Inductive reactance in Ohms.
    
    """
    l = normalize_inductance(l) if isinstance(l, str) else l
    f = normalize_frequency(f) if isinstance(f, str) else f
    return 2 * np.pi * f * l

@returns_unit("H")
def inductance_from_reactance(x: ResistanceOhm, f: FrequencyHz = 1000.0):
    """
    Compute the inductance (H) from an inductive reactance (Ω) at a given frequency f (Hz).

    Formula: X_L = 2 * pi * f * L => L = X_L / (2 * pi * f).

    Parameters
    ----------
    x : ResistanceOhm
        Inductive reactance in Ohms.
    f : FrequencyHz, optional
        Frequency in Hz. Default is 1000.0.

    Returns
    -------
    float
        Inductance in Henrys.
    
    """
    x = normalize_resistance(x) if isinstance(x, str) else x
    f = normalize_frequency(f) if isinstance(f, str) else f
    return x / (2 * np.pi * f)

@returns_unit("F")
def capacitance_from_reactance(x: ResistanceOhm, f: FrequencyHz = 1000.0):
    """
    Compute the capacitance (F) from a capacitive reactance (Ω) at a given
    
    frequency f (Hz).

    Formula: X_C = 1 / (2 * pi * f * C) => C = 1 / (2 * pi * f * X_C).

    Parameters
    ----------
    x : ResistanceOhm
        Capacitive reactance in Ohms.
    f : FrequencyHz, optional
        Frequency in Hz. Default is 1000.0.

    Returns
    -------
    float
        Capacitance in Farads.
    
    """
    x = normalize_resistance(x) if isinstance(x, str) else x
    f = normalize_frequency(f) if isinstance(f, str) else f
    return 1.0 / (2 * np.pi * f * x)

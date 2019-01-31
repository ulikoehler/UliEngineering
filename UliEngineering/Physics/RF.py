#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = ['quality_factor', 'resonant_impedance']

def quality_factor(frequency, bandwidth) -> Unit("V"):
    """
    Compute the quality factor of a resonant circuit
    from the frequency and the bandwidth:

    Q = frequency / bandwidth

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> quality_factor("8.000 MHz", "1 kHz")
    8000.0
    """
    frequency = normalize_numeric(frequency)
    bandwidth = normalize_numeric(bandwidth)
    return frequency / bandwidth

def resonant_impedance(L, C, Q=100.) -> Unit("V"):
    """
    Compute the resonant impedance of a resonant circuit

    R_res = sqrt(L / C) / Q

    The value returned is in Ohms.

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_impedance("100 uH", "10 nF", Q=30.0)
    3.333333333333333
    >>> format_value(resonant_impedance("100 uH", "10 nF", Q=30.0), "Ω")
    '3.33 Ω'
    """
    L = normalize_numeric(L)
    C = normalize_numeric(C)
    return np.sqrt(L / C) / Q

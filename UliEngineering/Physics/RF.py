#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = [
     'quality_factor', 'resonant_impedance', 'resonant_frequency',
     'resonant_inductance']

def quality_factor(frequency, bandwidth) -> Unit(""):
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

def resonant_impedance(L, C, Q=100.) -> Unit("Ω"):
    """
    Compute the resonant impedance of a resonant circuit

    R_res = sqrt(L / C) / Q

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_impedance("100 uH", "10 nF", Q=30.0)
    3.333333333333333
    >>> auto_format(resonant_impedance, "100 uH", "10 nF", Q=30.0)
    '3.33 Ω'
    """
    L = normalize_numeric(L)
    C = normalize_numeric(C)
    return np.sqrt(L / C) / Q

def resonant_frequency(L, C) -> Unit("Hz"):
    """
    Compute the resonant frequency of a resonant circuit
    given the inductance and capacitance.

    f = 1 / (2 * pi * sqrt(L * C))

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_frequency("100 uH", "10 nF")
    159154.94309189534
    >>> auto_format(resonant_frequency, "100 uH", "10 nF")
    '159 kHz'
    """
    L = normalize_numeric(L)
    C = normalize_numeric(C)
    return 1 / (2 * np.pi * np.sqrt(L * C))

def resonant_inductance(fres, C) -> Unit("H"):
    """
    Compute the inductance of a resonant circuit
    given the resonant frequency and its capacitance.

    L = 1 / (4 * pi² * fres² * C)

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_inductance("250 kHz", "10 nF")
    4.052847345693511e-05
    >>> auto_format(resonant_inductance, "250 kHz", "10 nF")
    '40.5 µH'
    """
    fres = normalize_numeric(fres)
    C = normalize_numeric(C)
    return 1 / (4 * np.pi**2 * fres**2 * C)

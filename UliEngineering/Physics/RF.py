#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np

__all__ = [
     'quality_factor', 'resonant_impedance', 'resonant_frequency',
     'resonant_inductance']

@returns_unit("")
@normalize_numeric_args
def quality_factor(frequency, bandwidth):
    """
    Compute the quality factor of a resonant circuit
    from the frequency and the bandwidth:

    Q = frequency / bandwidth

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> quality_factor("8.000 MHz", "1 kHz")
    8000.0
    """
    return frequency / bandwidth

@returns_unit("Ω")
@normalize_numeric_args
def resonant_impedance(L, C, Q=100.):
    """
    Compute the resonant impedance of a resonant circuit

    R_res = sqrt(L / C) / Q

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_impedance("100 uH", "10 nF", Q=30.0)
    3.333333333333333
    >>> auto_format(resonant_impedance, "100 uH", "10 nF", Q=30.0)
    '3.33 Ω'
    """
    return np.sqrt(L / C) / Q

@returns_unit("Hz")
@normalize_numeric_args
def resonant_frequency(L, C):
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
    return 1 / (2 * np.pi * np.sqrt(L * C))

@returns_unit("H")
@normalize_numeric_args
def resonant_inductance(fres, C):
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
    return 1 / (4 * np.pi**2 * fres**2 * C)

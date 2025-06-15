#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
import numpy as np
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = ["actual_noise", "noise_density"]
@normalize_numeric_args
@returns_unit("V")
def actual_noise(density, bandwith):
    """
    Compute the actual noise given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> autoFormat(actualNoise, "100 µV", "100 Hz")
    '1.00 mV'
    """
    return np.sqrt(bandwith) * density

@normalize_numeric_args
@returns_unit("V/√Hz")
def noise_density(actual_noise, bandwith):
    """
    Compute the noise density given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> formatValue(noiseDensity("1.0 mV", "100 Hz"), "V/√Hz")
    '100 μV/√Hz'
    """
    return actual_noise / np.sqrt(bandwith)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from UliEngineering.EngineerIO import normalize_numeric_args
from UliEngineering.Units import Unit
import numpy as np

__all__ = ["actual_noise", "noise_density"]

@normalize_numeric_args
def actual_noise(density, bandwith) -> Unit("V"):
    """
    Compute the actual noise given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> autoFormat(actualNoise, "100 µV", "100 Hz")
    '1.00 mV'
    """
    return np.sqrt(bandwith) * density

@normalize_numeric_args
def noise_density(actual_noise, bandwith) -> Unit("V/√Hz"):
    """
    Compute the noise density given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> formatValue(noiseDensity("1.0 mV", "100 Hz"), "V/√Hz")
    '100 μV/√Hz'
    """
    return actual_noise / np.sqrt(bandwith)

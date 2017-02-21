#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

def actualNoise(density, bandwith) -> Unit("V"):
    """
    Compute the actual noise given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> autoFormat(actualNoise, "100 µV", "100 Hz")
    '1.00 mV'
    """
    density = normalize_numeric(density)
    bandwith = normalize_numeric(bandwith)
    return np.sqrt(bandwith) * density

def noiseDensity(actual_noise, bandwith) -> Unit("V/√Hz"):
    """
    Compute the noise density given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> formatValue(noiseDensity("1.0 mV", "100 Hz"), "V/√Hz")
    '100 μV/√Hz'
    """
    actual_noise = normalize_numeric(actual_noise)
    bandwith = normalize_numeric(bandwith)
    return actual_noise / np.sqrt(bandwith)

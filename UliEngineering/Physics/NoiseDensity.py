#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from .Resistors import *
from .EngineerIO import *
import numpy as np

def actualNoise(density, bandwith):
    """
    Compute the actual noise given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> formatValue(actualNoise("100 µV", "100 Hz"), "V")
    '1.00 mV'
    """
    density, _ = normalizeEngineerInputIfStr(density)
    bandwith, _ = normalizeEngineerInputIfStr(bandwith)
    return np.sqrt(bandwith) * density

def noiseDensity(actual_noise, bandwith):
    """
    Compute the noise density given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> formatValue(noiseDensity("1.0 mV", "100 Hz"), "V/√Hz")
    '100 μV/√Hz'
    """
    actual_noise, _ = normalizeEngineerInputIfStr(actual_noise)
    bandwith, _ = normalizeEngineerInputIfStr(bandwith)
    return actual_noise / np.sqrt(bandwith)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

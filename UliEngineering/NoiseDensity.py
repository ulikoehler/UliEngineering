#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from .Resistors import *
from .EngineerIO import *
import numpy as np

def noise(density, bandwith):
    """
    Compute the actual noise given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> noise("100 mV", "100 Hz")
    1.0
    """
    if isinstance(density, str):
        density, _ = normalizeEngineerInput(density)
    if isinstance(bandwith, str):
        bandwith, _ = normalizeEngineerInput(bandwith)
    return np.sqrt(bandwith) * density

if __name__ == "__main__":
    import doctest
    doctest.testmod()

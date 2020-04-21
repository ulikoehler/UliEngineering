#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for acceleration
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np
import scipy.constants

g0 = scipy.constants.physical_constants['standard acceleration of gravity'][0]

__all__ = ["g_to_ms2", "ms2_to_g"]

def g_to_ms2(g) -> Unit("m/s²"):
    """
    Compute the acceleration in m/s² given the acceleration in g
    """
    g = normalize_numeric(g)
    return g * g0

def ms2_to_g(ms2) -> Unit("g"):
    """
    Compute the acceleration in g given the acceleration in m/s²
    """
    ms2 = normalize_numeric(ms2)
    return ms2 / g0

def centrifugal_acceleration(rpm) -> Unit("Hz"):
    """
    Compute the frequency associated with a period.

    Parameters
    ----------
    period : number or Engineer string or NumPy array-like
        The period in seconds
    """
    pass

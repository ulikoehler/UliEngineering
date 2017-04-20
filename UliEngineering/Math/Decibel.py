#!/usr/bin/env python3
"""
Utilities for FFT computation and visualization
"""
import numpy as np
from UliEngineering.EngineerIO import normalize_numeric

__all__ = ["ratio_to_db_field", "ratio_to_db_power", "value_to_db_field", "value_to_db_power"]

def _safe_log10(v):
    """
    Log10 with negative input => -np.inf
    """
    if isinstance(v, np.ndarray):
        v[v < 0] = 0
    else:
        if v < 0:
            return -np.inf
    return np.log10(v)

def ratio_to_db_field(ratio):
    """
    Convert a given ratio to a decibel value for field quantities

    dBs = 20 * log10(ratio)

    Ratio 2 ~= 6 dB
    Ratio 1 ~= 0 dB
    Ratio 0.5 ~= -6 dB

    Returns -np.inf for negative values
    """
    return 20 * _safe_log10(ratio)

def ratio_to_db_power(ratio):
    """
    Convert a given ratio to a decibel value for power quantities.

    dB = 20 * log10(ratio)

    Ratio 2 ~= 3 dB
    Ratio 1 ~= 0 dB
    Ratio 0.5 ~= -3 dB

    Returns -np.inf for negative values
    """
    return 10 * _safe_log10(ratio)

def value_to_db_field(v, v0):
    """
    Convert a given field quantity v to dB
    in reference to a given reference quantity v0.

    dB = 20 * log10(v / v0)

    Can also interpret engineering strings

    Returns -np.inf for negative values
    """
    v = normalize_numeric(v)
    v0 = normalize_numeric(v0)
    return ratio_to_db_field(v / v0)

def value_to_db_power(p, p0):
    """
    Convert a given field quantity v to dB
    in reference to a given reference quantity v0.

    dB = 20 * log10(v / v0)

    Can also interpret engineering strings

    Returns -np.inf for negative values
    """
    p = normalize_numeric(p)
    p0 = normalize_numeric(p0)
    return ratio_to_db_power(p / p0)

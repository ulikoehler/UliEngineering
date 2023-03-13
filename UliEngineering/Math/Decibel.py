#!/usr/bin/env python3
"""
Utilities for FFT computation and visualization
"""
import numpy as np
from UliEngineering.Units import Unit
from UliEngineering.EngineerIO import normalize_numeric

__all__ = [
    "dBFactor",
    "ratio_to_dB",
    "dB_to_ratio",
    "value_to_dB",
    "dB_to_value",
    "voltage_to_dBuV",
    "dBuV_to_voltage",
    "power_to_dBm"
]

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

class dBFactor:
    """Pre-set values for factors"""
    Power = 10.
    Field = 20.

def ratio_to_dB(ratio, factor=dBFactor.Field) -> Unit("dB"):
    """
    Convert a given ratio to a decibel value.
    For power quantities, set factor=dBFactor.Power
    For field quantities, set factor=dBFactor.Field

    dB = [factor] * log10(ratio)

    Returns -np.inf for negative values
    """
    ratio = normalize_numeric(ratio)
    return factor * _safe_log10(ratio)

def dB_to_ratio(dB, factor=dBFactor.Field):
    """
    Convert a given ratio from a decibel value to the underlying quantity.
    The result is returned as a ratio to the 0 dB value.
    
    For power quantities, set factor=dBFactor.Power
    For field quantities, set factor=dBFactor.Field
    """
    dB = normalize_numeric(dB)
    return 10**(dB/factor)

def value_to_dB(v, v0=1.0, factor=dBFactor.Field) -> Unit("dB"):
    """
    Convert a given quantity [v] to dB, with 0dB being [v0].
    
    Returns -np.inf for negative values
    """
    v = normalize_numeric(v)
    v0 = normalize_numeric(v0)
    return ratio_to_dB(v / v0, factor=factor)

def dB_to_value(dB, v0=1.0, factor=dBFactor.Field):
    """
    Convert a given decibel value [dB] to dB, with 0 dB being [v0].
    
    Returns -np.inf for negative values
    """
    dB = normalize_numeric(dB)
    v0 = normalize_numeric(v0)
    return dB_to_ratio(dB, factor=factor) * v0

# Utility functions
def voltage_to_dBuV(v) -> Unit("dBµV"):
    """
    Represent a voltage as dB microvolts.

    Also see the online calculator at
    https://techoverflow.net/2019/07/29/volts-to-db%c2%b5v-online-calculator-ampamp-python-code/
    """
    return value_to_dB(v, 1e-6, factor=dBFactor.Field)

def dBuV_to_voltage(v) -> Unit("V"):
    """
    Represent a dB microvolt voltage in volt.

    Also see the online calculator at
    https://techoverflow.net/2019/07/28/db%c2%b5v-to-volts-online-calculator-python-code/
    """
    return dB_to_ratio(v, factor=dBFactor.Field) * 1e-6

def power_to_dBm(v) -> Unit("dBm"):
    """
    Represent a power in Watts as dB milliwatts.
    """
    return value_to_dB(v, 1e-3, factor=dBFactor.Power)

#def dbM_to_power(v) -> Unit("dBµV"):
#    """
#    Represent a power in Watts as dB milliwatts.
#    """
#    return db_power_to_ratio(v) * 1e-6
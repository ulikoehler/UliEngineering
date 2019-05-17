#!/usr/bin/env python3
"""
Utilities to compute the power factor
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = ["power_factor_by_phase_angle"]

def power_factor_by_phase_angle(angle="10°", unit="degrees") -> Unit(""):
    """
    Compute the power factor given the phase angle between current and voltage
    This approach only returns the correct power factor if current and voltage
    are both sinusoidal.

    Keyword arguments:
    ------------------
    angle : number or Engineer strings
        The phase angle between current and voltage.
    unit : "degrees", "deg" or "radians", "rad", "radiant"
        The unit to interpret angle as
    """
    angle = normalize_numeric(angle)
    if unit in ["degrees", "deg"]:
        angle = np.deg2rad(angle)
    elif unit in ["radians", "rad", "radiant"]:
        pass # No need to convert
    else:
        raise ValueError(f"Angle unit '{unit}' is unknown, use 'degrees' or 'radians'!")
    return np.cos(angle)
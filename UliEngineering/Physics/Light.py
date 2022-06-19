#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = ["lumen_to_candela_by_apex_angle"]

def lumen_to_candela_by_apex_angle(flux, angle) -> Unit("cd"):
    """
    Compute the luminous intensity from the luminous flux,
    assuming that the flux of <flux> is distributed equally around
    a cone with apex angle <angle>.

    Keyword parameters
    ------------------
    flux : value, engineer string or NumPy array
        The luminous flux in Lux.
    angle : value, engineer string or NumPy array
        The apex angle of the emission cone, in degrees
        For many LEDs, this is 

    >>> autoFormat(lumen_to_candela_by_apex_angle, "25 lm", "120°")
    '7.96 cd'
    """
    flux = normalize_numeric(flux)
    angle = normalize_numeric(angle)
    solid_angle = 2*np.pi*(1.-np.cos(np.deg2rad(angle)/2.0))
    return flux / solid_angle

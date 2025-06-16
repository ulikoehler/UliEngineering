#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate inductors
"""

__all__ = ["ideal_inductor_current_change_rate"]

import numpy as np
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

@returns_unit("A/s")
@normalize_numeric_args
def ideal_inductor_current_change_rate(inductance, voltage):
    """
    Compute the rise or fall rate of current in an ideal inductor,
    if there's [voltage] across it.
    
    Parameters
    ----------
    inductance: number or Engineer string
        The inductance in Henrys
    voltage: number or Engineer string
        The voltage across the inductor
    """
    return np.divide(voltage, inductance)  # Returns inf when inductance is zero
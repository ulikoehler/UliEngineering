#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate inductors
"""
import numpy as np
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit

__all__ = ["ideal_inductor_current_change_rate"]

def ideal_inductor_current_change_rate(inductance, voltage) -> Unit("A/s"):
    """
    Compute the rise or fall rate of current in an ideal inductor,
    if there's [voltage] across it.
    
    Parameters
    ----------
    inductance: number or Engineer string
        The inductance in Henrys
    vsupply: number or Engineer string
        The voltage across the inductor
    """
    inductance = normalize_numeric(inductance)
    voltage = normalize_numeric(voltage)
    return voltage / inductance
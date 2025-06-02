#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate inductors
"""
from UliEngineering.EngineerIO import normalize_numeric_args
from UliEngineering.Units import Unit

__all__ = ["ideal_inductor_current_change_rate"]

@normalize_numeric_args
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
    return voltage / inductance
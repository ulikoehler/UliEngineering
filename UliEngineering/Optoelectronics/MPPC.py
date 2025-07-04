#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Multi-Pixel photon counters (MPPCs)
"""

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "pixel_capacitance_from_terminal_capacitance"
]

@normalize_numeric_args
@returns_unit("F")
def pixel_capacitance_from_terminal_capacitance(terminal_capacitance="900pF", npixels=14331):
    """
    Estimate a MPPC's individual pixel's capacitance from the terminal capacitance.
    
    Typically, this overestimates the capacitance because the case & trace capacitance is included
    in the terminal capacitance. The overestimation effect is particularly large for MPPCs with very small
    pixels such as 15μm.
    
    This method is outlined as alternate method [by Hamamatsu](https://hub.hamamatsu.com/us/en/technical-notes/mppc-sipms/a-technical-guide-to-silicon-photomutlipliers-MPPC-Section-3.html)
    """
    return terminal_capacitance / npixels

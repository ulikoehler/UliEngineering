#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for Multi-Pixel photon counters (MPPC utilities)."""

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Types import NormalizableArgument
from .Electronics.Capacitors import normalize_capacitance, CapacitanceFarad

__all__ = [
    "pixel_capacitance_from_terminal_capacitance"
]

@returns_unit("F")
def pixel_capacitance_from_terminal_capacitance(terminal_capacitance: CapacitanceFarad = "900pF", npixels: NormalizableArgument = 14331):
    """
    Estimate a MPPC's individual pixel's capacitance from the terminal capacitance.

    Typically, this overestimates the capacitance because the case & trace capacitance is included
    in the terminal capacitance. The overestimation effect is particularly large for MPPCs with very small
    pixels such as 15μm.

    This method is outlined as alternate method [by Hamamatsu](https://hub.hamamatsu.com/us/en/technical-notes/mppc-sipms/a-technical-guide-to-silicon-photomutlipliers-MPPC-Section-3.html)
    """
    terminal_capacitance = normalize_capacitance(terminal_capacitance) if isinstance(terminal_capacitance, str) else terminal_capacitance
    npixels = normalize_numeric(npixels) if isinstance(npixels, str) else npixels
    return terminal_capacitance / npixels

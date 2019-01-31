#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computations related to noise density
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

def quality_factor(frequency, bandwidth) -> Unit("V"):
    """
    Compute the quality factor of a resonant circuit
    from the frequency and the bandwidth

    >>> quality_factor("8.000 MHz", "1 kHz")
    8000.0
    """
    frequency = normalize_numeric(frequency)
    bandwidth = normalize_numeric(bandwidth)
    return frequency / bandwidth

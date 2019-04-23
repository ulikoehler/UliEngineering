#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for frequencies
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = ["frequency_to_period", "period_to_frequency"]


def frequency_to_period(frequency) -> Unit("s"):
    """
    Compute the period associated with a frequency.

    Parameters
    ----------
    frequency : number or Engineer string or NumPy array-like
        The frequency in Hz
    """
    # Normalize inputs
    frequency = normalize_numeric(frequency)
    # Compute resistance
    return 1./frequency

def period_to_frequency(period) -> Unit("Hz"):
    """
    Compute the frequency associated with a period.

    Parameters
    ----------
    period : number or Engineer string or NumPy array-like
        The period in seconds
    """
    # Normalize inputs
    period = normalize_numeric(period)
    # Compute resistance
    return 1./period

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for frequencies
"""
from UliEngineering.EngineerIO import normalize_numeric_args
from UliEngineering.Units import Unit

__all__ = ["frequency_to_period", "period_to_frequency"]

@normalize_numeric_args
def frequency_to_period(frequency) -> Unit("s"):
    """
    Compute the period associated with a frequency.

    Parameters
    ----------
    frequency : number or Engineer string or NumPy array-like
        The frequency in Hz
    """
    return 1./frequency

@normalize_numeric_args
def period_to_frequency(period) -> Unit("Hz"):
    """
    Compute the frequency associated with a period.

    Parameters
    ----------
    period : number or Engineer string or NumPy array-like
        The period in seconds
    """
    return 1./period

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for frequencies
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = ["frequency_to_period", "period_to_frequency"]

@returns_unit("s")
@normalize_numeric_args
def frequency_to_period(frequency):
    """
    Compute the period associated with a frequency.

    Parameters
    ----------
    frequency : number or Engineer string or NumPy array-like
        The frequency in Hz
    """
    return 1./frequency

@returns_unit("Hz")
@normalize_numeric_args
def period_to_frequency(period):
    """
    Compute the frequency associated with a period.

    Parameters
    ----------
    period : number or Engineer string or NumPy array-like
        The period in seconds
    """
    return 1./period

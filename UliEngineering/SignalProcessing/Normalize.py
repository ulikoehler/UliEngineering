#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for normalizing signals
"""
import numpy as np
from collections import namedtuple
import scipy.signal
from .Utils import peak_to_peak

__all__ = ["normalize_max", "center_to_zero", "normalize_minmax"]

NormalizationResult = namedtuple("NormalizationResult", ["data", "factor", "offset"])

def normalize_max(signal):
    """
    Normalize signal by dividing by its max value.
    Does not perform any offset adjustment

    Returns
    -------
    A NormalizationResult() object.
    Use .data to access the data
    Use .factor to access the factor that signal was divided by
    Use .offset to access the offset that was subtracted from signal
    """
    mx = np.max(signal)
    return NormalizationResult(signal / mx, mx, 0.)

def normalize_minmax(signal):
    """
    Normalize signal by setting its lowest value
    to 0.0 and its highest value to 1.0,
    keeping all other values

    Returns
    -------
    A NormalizationResult() object.
    Use .data to access the data
    Use .factor to access the factor that signal was divided by
    Use .offset to access the offset that was subtracted from signal
    """
    mi = np.min(signal)
    mx = np.max(signal)
    factor = mx - mi
    return NormalizationResult((signal - mi) / factor, factor, mi)

def center_to_zero(signal):
    """
    Normalize signal by subtracting its mean
    Does not perform any factor normalization

    Returns
    -------
    A NormalizationResult() object.
    Use .data to access the data
    Use .factor to access the factor that signal was divided by
    Use .offset to access the offset that was subtracted from signal
    """
    mn = np.mean(signal)
    return NormalizationResult(signal - mn, 1., mn)
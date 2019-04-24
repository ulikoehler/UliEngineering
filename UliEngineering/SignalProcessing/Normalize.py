#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for normalizing signals
"""
import numpy as np
from collections import namedtuple
import scipy.signal
from .Utils import peak_to_peak

__all__ = ["normalize_max", "center_to_zero", "normalize_minmax", "normalize_plusminus_peak"]

NormalizationResult = namedtuple("NormalizationResult", ["data", "factor", "offset"])

def normalize_max(signal):
    """
    Normalize signal by dividing by its max value.
    Does not perform any offset adjustment.

    This approach works well for data that is guaranteed to be
    positive and if no offset adjustment is desired.

    In case signal has a max of <= 0.0, signal is returned.
    For a similar function that also limits the minimum value,
    see normalize_plusminus_peak.

    Returns
    -------
    A NormalizationResult() object.
    Use .data to access the data
    Use .factor to access the factor that signal was divided by
    Use .offset to access the offset that was subtracted from signal
    """
    if len(signal) == 0:
        return NormalizationResult([], 1., 0.)
    mx = np.max(signal)
    # Avoid divide by zero
    if mx <= 0.:
        return NormalizationResult(signal, 1., 0.)
    return NormalizationResult(signal / mx, mx, 0.)

def normalize_minmax(signal):
    """
    Normalize signal by setting its lowest value
    to 0.0 and its highest value to 1.0,
    keeping all other values.

    If signal consists of only zeros, no factor
    normalization is applied.

    Returns
    -------
    A NormalizationResult() object.
    Use .data to access the data
    Use .factor to access the factor that signal was divided by
    Use .offset to access the offset that was subtracted from signal
    """
    if len(signal) == 0:
        return NormalizationResult([], 1., 0.)
    mi = np.min(signal)
    mx = np.max(signal)
    factor = mx - mi
    if factor == 0.0:
        factor = 1.0
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


def normalize_plusminus_peak(signal):
    """
    Center a signal to zero and normalize so that
        - np.max(result) is <= 1.0
        - np.min(result) is <= 1.0

    Returns
    -------
    A NormalizationResult() object.
    Use .data to access the data
    Use .factor to access the factor that signal was divided by
    Use .offset to access the offset that was subtracted from signal
    """
    norm_res = center_to_zero(signal)
    mi = np.min(norm_res.data)
    mx = np.max(norm_res.data)
    factor = max(mi, mx)
    return NormalizationResult(norm_res / factor, factor, norm_res.offset)



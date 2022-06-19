#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for correlating a dataset with itself and other datasets
Mainly built for 1D signal analysis.
Might or might not work for higher-dimensional data.
"""
import scipy.signal

__all__ = ["autocorrelate"]

def autocorrelate(signal):
    """
    Auto-correlate a signal with itself.

    Based on the fast FFT convolution using
    scipy.signal.fftconvolve.
    """
    return scipy.signal.fftconvolve(signal, signal[::-1], mode='full')

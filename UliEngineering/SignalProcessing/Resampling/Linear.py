#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear interpolation resampler
"""
import collections

"""Represents the shape of a resampling operation"""
ResamplingShape = collections.namedtuple("ResamplingShape", ["tmin", "tmax", "n", "samplerate", "time_factor"])

def computeResamplingShape(t, new_samplerate, assume_sorted=True, time_factor=1e6):
    """
    """
    if len(t) == 0:
        raise ValueError("Empty time array given - can not perform any resampling")
    if len(t) == 1:
        raise ValueError("Time array has only one value - can not perform any resampling")
    # Compute time corners
    sample_tdiff = time_factor / samplerate
    startt, endt = t[0], t[-1] if assume_sorted else np.min(t), np.max(t)
    if endt - startt < sample_tdiff:
        raise ValueError("The time delta is smaller than a single sample - can not perform resampling")
    
    return ResamplingShape(startt, endt, samplerate, time_factor)

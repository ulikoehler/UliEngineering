#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for selecting and finding specific attributes in datasets
"""
import numpy as np
from scipy.interpolate import splrep, splev

__all__ = ["resample_discard", "BSplineResampler"]


class BSplineResampler(object):
    """
    Arbitrary resampler using scipy bsplines
    """
    def __init__(self, fx, fy, time_factor=1e6, prefilt=None):
        if prefilt is not None:
            fy = filt(fy)
        self.fx = fx
        self.fy = fy
        self.time_factor = time_factor
        # Generate bspline
        self.tck = splrep(fx, fy)

    @property
    def samplerate_stddev(self):
        "Get the variance of the differential sample rate"
        return np.stddev(np.diff(self.fx))

    @property
    def actual_samplerate(self):
        return self.time_factor / np.mean(np.diff())

    def resample(self, samples):
        "Resample with a specific sample array. Alle samples must lie within the original samplespace"
        y = splev(samples, self.tck)
        return samples, y

    def resample_to(self, samplerate):
        "Resample to a specific samplerate. Returns fx, fy"
        numSamples = (self.fx[-1] - self.fx[0]) * samplerate / self.time_factor
        return self.resample(np.linspace(self.fx[0], self.fx[-1], numSamples))


def resample_discard(arr, divisor, ofs=0):
    """
    Resample with an integral divisor, discarding all other samples
    """
    return arr[ofs::divisor]

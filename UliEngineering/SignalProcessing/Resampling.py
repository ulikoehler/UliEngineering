#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for selecting and finding specific attributes in datasets
"""
import numpy as np
from scipy.interpolate import splrep, splev

__all__ = ["resample_discard", "BSplineResampler", "ResampledFilteredXYView",
           "ResampledFilteredView", "ResampledFilteredViewYOnlyDecorator"]


class BSplineResampler(object):
    """
    Arbitrary resampler using scipy bsplines
    """
    def __init__(self, fx, fy, time_factor=1e6, prefilt=None):
        if prefilt is not None:
            fy = prefilt(fy)
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
    Resample with an integral divisor, discarding all other samples.
    Returns a view of the data.
    """
    return arr[ofs::divisor]


class ResampledFilteredXYView(object):
    """
    Lazy resampling 1D array view that can be used e.g. as input for a chunk generator.
    Note that this class will return an ill-sized slice in most cases, as
    the.
    The user must therefore ensure that downstream functions appropriately deal with those slices.
    One way of doing that is to oversize the slices requested by the caller and ensure they are properly
    trimmed downstream.
    
    This class returns (xslice, yslice) on slice. See ResampledFilteredView for a wrapper that only
    returns the Y view.
    """
    def __init__(self, fx, fy, target_samplerate, time_factor=1e6, filt=None):
        self.fx = fx
        self.fy = fy
        self.samplerate = target_samplerate
        self.filt = filt
        self.time_factor = time_factor

    def getslice(self, start, stop, step):
        xslice = self.fx[start:stop:step]
        yslice = self.fy[start:stop:step]
        return BSplineResampler(xslice, yslice, time_factor=time_factor,
                                prefile=self.filt).resample_to(self.target_samplerate)


class ResampledFilteredView(ResampledFilteredXYView):
    "Like ResampledFilteredXYView, but only returns y values on slice"
    def getslice(self, start, stop, step):
        _, y = super(self.__class__, self).getslice(start, stop, step)
        return y


class ResampledFilteredViewYOnlyDecorator(object):
    """
    Like ResampledFilteredXYView, but only returns y values on slice.
    Decorator version of ResampledFilteredView.

    Usage example:
        rv = ResampledFilteredXYView(...)
        rv_yonly = ResampledFilteredViewYOnlyDecorator(rv)
    """
    def __init__(self, other):
        self.other = other

    def getslice(self, start, stop, step):
        _, y = self.other.getslice(start, stop, step)
        return y

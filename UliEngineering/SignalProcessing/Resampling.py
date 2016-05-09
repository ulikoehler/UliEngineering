#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for selecting and finding specific attributes in datasets
"""
import math
import functools
import numpy as np
import bisect
import concurrent.futures
from scipy.interpolate import splrep, splev, UnivariateSpline
from toolz.functoolz import identity
from UliEngineering.Utils.Concurrency import new_thread_executor
from .Utils import LinRange

__all__ = ["resample_discard", "BSplineResampler", "ResampledFilteredXYView",
           "ResampledFilteredView", "ResampledFilteredViewYOnlyDecorator",
           "parallel_resample"]

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
        "Get the variance of the differential source sample rate"
        return np.stddev(np.reciprocate(np.diff(self.fx)))

    @property
    def actual_samplerate(self):
        return self.time_factor / np.mean(np.diff())

    def resample(self, samples):
        "Resample with a specific sample array. Alle samples must lie within the original samplespace"
        y = splev(samples, self.tck, ext=2) # ext: raise ValueError on out of bounds access
        return samples, y

    def resample_to(self, samplerate):
        "Resample to a specific samplerate. Returns fx, fy"
        num_samples = (self.fx[-1] - self.fx[0]) * samplerate / self.time_factor
        return self.resample(np.linspace(self.fx[0], self.fx[-1], num_samples))


def resample_discard(arr, divisor, ofs=0):
    """
    Resample with an integral divisor, discarding all other samples.
    Returns a view of the data.
    """
    return arr[ofs::divisor]


class ResampledFilteredXYView(object):
    """
    Lazy resampling 1D array view that can be used e.g. as input for a chunk generator.
    Note that this class will generate an ill-sized last slice in most cases.
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

    def __getitem__(self, key):
        if isinstance(key, slice):
            xslice = self.fx.__getitem__(key)
            yslice = self.fy.__getitem__(key)
            resampler = BSplineResampler(xslice, yslice, time_factor=self.time_factor,
                                         prefilt=self.filt)
            return resampler.resample_to(self.samplerate)
        elif isinstance(key, int):
            raise TypeError("ResampledFilteredXYView can only be sliced with slice indices, not single numbers")
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(type(key))) 

    @property
    def shape(self):
        return self.fy.shape

class _XViewDecorator(object):
    "This class is used in ResampledFilteredView to provide the time object"
    def __init__(self, delegate):
        self.delegate = delegate

    def __getitem__(self, key):
        if isinstance(key, slice):
            x, _ = ResampledFilteredXYView.__getitem__(self.delegate, key)
            return x
        elif isinstance(key, int):
            raise TypeError("ResampledFilteredView.time can only be sliced with slice indices, not single numbers")
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(type(key)))

class ResampledFilteredView(ResampledFilteredXYView):
    "Like ResampledFilteredXYView, but only returns y values on slice"

    @property
    def time(self):
        return _XViewDecorator(self)

    def __getitem__(self, key):
        if isinstance(key, slice):
            _, y = super().__getitem__(key)
            return y
        elif isinstance(key, int):
            raise TypeError("ResampledFilteredView can only be sliced with slice indices, not single numbers")
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(type(key))) 


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

    def __getitem__(self, key):
        if isinstance(key, slice):
            _, y = self.other.__getitem__(key)
            return y
        elif isinstance(key, int):
            raise TypeError("ResampledFilteredView can only be sliced with slice indices, not single numbers")
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(type(key)))

    @property
    def shape(self):
        return self.other.shape


def __compute_new_samples(t, new_samplerate, assume_sorted=True, time_factor=1e6):
    """
    Compute a lazy LinRange of new sample times when resampling
    a given time array with a new samplerate.
    Also checks if the given time array is large enough for proper resampling.
    """
    if len(t) == 0:
        raise ValueError("Empty time array given - can not perform any resampling")
    if len(t) == 1:
        raise ValueError("Time array has only one value - can not perform any resampling")
    # Compute time corners
    sample_tdelta = time_factor / new_samplerate
    startt, endt = (t[0], t[-1]) if assume_sorted else (np.min(t), np.max(t))
    tdelta = endt - startt
    if tdelta < sample_tdelta:
        raise ValueError("The time delta is smaller than a single sample - can not perform resampling")
    return LinRange.range(startt, endt, sample_tdelta)


def __parallel_resample_worker(torig, tnew, y, out, i, degree, chunksize, ovp_size, prefilter):
    # Find the time range in the target time
    t_target = tnew[i:i + chunksize]
    # Find the time range in the source time
    srcstart = bisect.bisect_left(torig, t_target[0])
    srcend = bisect.bisect_right(torig, t_target[1])
    # Compute start and end index with overprovisioning
    # This might be out of range of the src array but bisect will ignore that
    srcstart_ovp = max(0, srcstart - ovp_size)  # Must not get negative indices
    srcend_ovp = srcend - ovp_size
    # Compute source slices
    tsrc_chunk = torig[srcstart_ovp:srcend_ovp]
    ysrc_chunk = y[srcstart_ovp:srcend_ovp]

    # Perform prefilter
    if prefilter is not None:
        tsrc_chunk, ysrc_chunk = prefilter(tsrc_chunk, ysrc_chunk)

    # Compute interpolating spline (might also be piecewise linear)...
    spline = UnivariateSpline(tsrc_chunk, ysrc_chunk, k=degree)
    # ... and evaluate
    out[i:i + chunksize] = spline(t_target)


def parallel_resample(t, y, new_samplerate, out=None, prefilter=None,
                      executor=None, time_factor=1e6, degree=1,
                      chunksize=10000, overprovisioning_factor=0.01):
    """
    A resampler that uses scipy.interpolate.UnivariateSpline but splits the
    input into chunks that can be processed. The chunksize is applied to the output timebase.

    Applies an optional prefilter to the source data while resampling. If the timebase of
    the source data is off significantly, this might produce unexpected results.
    The prefilter must be a reentrant functor that takes (t, x) data and returns
    a (t, x) tuple. The returned tuple can be of arbitrary size (assuming t and x
    have the same length) but its t range must include the t range that is being interpolated.
    Note that the prefilter is performed after overprovisioning, so setting a higher
    overprovisioning factor (see below) might help dealing with prefilters that
    return too small arrays, however at the start and the end of the source array,
    no overprovisioning values can be added.

    The input x array is assumed to be sorted. This function will also take mmapped input
    and output arrays.
    If the output array is not given, it is automatically allocated with the correct size.

    The chunk workers are executed in parallel in a concurrent.futures thread pool.

    In order to account for , an overprovisioning factor
    can be provided so that a fraction of the chunksize is added at both ends of
    the source chunk. This is used for higher-degree splines that perform better
    when not interpolating right on the edges of the source value space.
    A overprovisioning factor of 0.01 means that 1% of the chunksize is added on the left
    and 1% is added on the right. At the borders, only what's available is added
    to the array.

    Return the output array.
    """
    new_t = __compute_new_samples(t, new_samplerate, time_factor=time_factor)
    # Lazily compute the new timespan
    if out is None:
        out = np.zeros(len(new_t))
    if executor is None:
        executor = new_thread_executor()
    ovp_size = int(math.floor(overprovisioning_factor * chunksize))
    # How many chunks do we have to process?
    numchunks = len(new_t) // chunksize

    # Bind constant arguments
    f = functools.partial(__parallel_resample_worker, torig=t, tnew=new_t,
                          y=y, out=out, degree=degree, chunksize=chunksize,
                          ovp_size=ovp_size, prefilter=prefilter)

    futures = [executor.submit(f, i=i) for i in range(numchunks)]
    # Wait for futures to finish
    concurrent.futures.wait(futures)

    return out

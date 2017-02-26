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
from scipy.interpolate import UnivariateSpline
from UliEngineering.Utils.Concurrency import new_thread_executor
from .Utils import LinRange

__all__ = ["resample_discard", "resampled_timespace",
           "parallel_resample"]


def resample_discard(arr, divisor, ofs=0):
    """
    Resample with an integral divisor, discarding all other samples.
    Returns a view of the data.
    Very fast as this doesn't need to read the data.
    """
    return arr[ofs::divisor]

def resampled_timespace(t, new_samplerate, assume_sorted=True, time_factor=1e6):
    """
    Compute the new timespace after resampling a source timestamp array
    (not neccessarily lazy)


    Parameters
    ----------
    t : numpy array-like
        The source timestamps
    new_samplerate : float
        The new datarate in Hz
    assume_sorted : bool
        If this is True, the code assumes the source
        timestamp array is monotonically increasing, i.e.
        the lowest timestamp comes first and the highest last.
        If this is False, the code determines
        the min/max value by reading the entire array.
    time_factor : float
        Defines what timestamps in the source (and result)
        array means. This is required to interpret new_samplerate.
        If time_factor=1e6, it means that a difference of 1.0
        in two timestamps means a difference of 1/1e6 seconds.
    
    Returns
    -------
    A LinSpace() (acts like a numpy array but doesn't consume any memory)
    that represents the new timespace
    news
    """
    if len(t) == 0:
        raise ValueError("Empty time array given - can not perform any resampling")
    if len(t) == 1:
        raise ValueError("Time array has only one value - can not perform any resampling")
    # Compute time endpoints
    dst_tdelta = time_factor / new_samplerate
    startt, endt = (t[0], t[-1]) if assume_sorted else (np.min(t), np.max(t))
    src_tdelta = endt - startt
    if src_tdelta < dst_tdelta:
        raise ValueError("The time delta is smaller than a single sample - can not perform resampling")
    # Use a lazy linrange to represent time interval
    return LinRange.range(startt, endt, dst_tdelta)


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
    new_t = resampled_timespace(t, new_samplerate, time_factor=time_factor)
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

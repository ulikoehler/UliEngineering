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
import scipy.interpolate
from UliEngineering.Utils.Concurrency import QueuedThreadExecutor
from .Utils import LinRange

__all__ = ["resample_discard", "resampled_timespace",
           "parallel_resample", "signal_samplerate",
           "serial_resample"]

def signal_samplerate(t, ignore_percentile=10, mean_method=np.mean):
    """
    Compute the samplerate of a signal
    using a quantile-based method to exclude
    outliers (in the time delta domain) and
    computes the by 1 / mean
    
    Using a low ignore_percentile value is only
    desirable if the dataset is small and therefore
    does not average properly due to lack of samples.
    In most cases, using a high ignore percentile
    like 10 is recommended.
    
    Returns a float (samplerate) [1/s]

    Parameters
    ----------
    t : numpy array of datetime64 type
        Timestamps associated with the signal
    ignore_percentile : number
        This percentile of outliers is ignored
        for the mean calculation at both the top
        and the bottom end.
        "5" means considering the 5th...95th percentile
        for averaging.
    mean_method : unary function
        Used to compute the mean after excluding outliers.
        Except for special usecases, arithmetic mean (np.mean)
        is recommended.
    """
    tdelta = np.diff(t)
    above = np.percentile(tdelta, ignore_percentile)
    below = np.percentile(tdelta, 100 - ignore_percentile)
    filtered = tdelta[np.logical_and(tdelta >= above, tdelta <= below)]
    # Filtered is too small if the sample periods are too uniform in the array
    if len(filtered) < 0.1 * len(tdelta):
        filtered = tdelta
    mean_sample_period = mean_method(filtered)
    mean_sample_period = mean_sample_period.astype("timedelta64[ns]").astype(np.int64)
    return 1e9 / mean_sample_period # 1e9 : nanoseconds

def resample_discard(arr, divisor, ofs=0):
    """
    Resample with an integral divisor, discarding all other samples.
    Returns a view of the data.
    Very fast as this doesn't need to read the data.
    """
    return arr[ofs::divisor]

def resampled_timespace(t, new_samplerate, assume_sorted=True, time_factor=1e6):
    """
    Compute the new timespace after resampling a input timestamp array
    (not neccessarily lazy)

    Parameters
    ----------
    t : numpy array-like
        The source timestamps.
        If these are numbers, you must supply time_factor to
        specify the resolution of the number.
        If they are 
    new_samplerate : float
        The new datarate in Hz
    assume_sorted : bool
        If this is True, the code assumes the source
        timestamp array is monotonically increasing, i.e.
        the lowest timestamp comes first and the highest last.
        If this is False, the code determines
        the min/max value by reading the entire array.
    time_factor : float
        Ignored if t is of dtype datetime64
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
    # Handle numpy datetime64 input
    if "datetime64" in t.dtype.name:
        t = t.astype('datetime64[ns]').astype(np.int64)
        time_factor = 1e9
    # Compute time endpoints
    dst_tdelta = time_factor / new_samplerate
    startt, endt = (t[0], t[-1]) if assume_sorted else (np.min(t), np.max(t))
    src_tdelta = endt - startt
    if src_tdelta < dst_tdelta:
        raise ValueError("The time delta is smaller than a single sample - can not perform resampling")
    # Use a lazy linrange to represent time interval
    return LinRange.range(startt, endt, dst_tdelta)


def __parallel_resample_worker(torig, tnew, y, out, i, chunksize, ovp_size, prefilter, fitkind):
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
    fit = scipy.interpolate.interp1d(tsrc_chunk, ysrc_chunk, fitkind=fitkind)
    # ... and evaluate
    out[i:i + chunksize] = fit(t_target)


def serial_resample(t, y, new_samplerate, out=None, prefilter=None,
                      time_factor=1e6,
                      fitkind='linear', chunksize=10000,
                      overprovisioning_factor=0.01):
    """
    A resampler that uses scipy.interpolate.interp1d but splits the
    input into chunks that can be processed.
    The chunksize is applied to the output timebase.

    The input x array is assumed to be sorted, facilitating binary search.
    If the output array is not given, it is automatically allocated with the correct size.

    The chunk workers are executed in parallel in a concurrent.futures thread pool.

    In order to account for vector end effects, an overprovisioning factor
    can be provided so that a fraction of the chunksize is added at both ends of
    the source chunk.
    This
    A overprovisioning factor of 0.01 means that 1% of the chunksize is added on the left
    and 1% is added on the right. This does not affect leftmost and rightmost
    border of the input array.

    Returns the output array.

    Applies an optional prefilter to the input data while resampling. If the timebase of
    the input data is off significantly, this might produce unexpected results.
    The prefilter must be a reentrant functor that takes (t, x) data and returns
    a (t, x) tuple. The returned tuple can be of arbitrary size (assuming t and x
    have the same length) but its t range must include the t range that is being interpolated.
    Note that the prefilter is performed after overprovisioning, so setting a higher
    overprovisioning factor (see below) might help dealing with prefilters that
    return too small arrays, however at the start and the end of the input array,
    no overprovisioning values can be added.
    """
    new_t = resampled_timespace(t, new_samplerate, time_factor=time_factor)
    # Lazily compute the new timespan
    if out is None:
        out = np.zeros(len(new_t))
    ovp_size = int(math.floor(overprovisioning_factor * chunksize))
    # How many chunks do we have to process?
    for i in range(len(new_t) // chunksize):
        __parallel_resample_worker(i=i, orig=t, tnew=new_t,
            y=y, out=out, chunksize=chunksize,
            ovp_size=ovp_size, prefilter=prefilter,
            fitkind=fitkind)

    return out


def parallel_resample(t, y, new_samplerate, out=None, prefilter=None,
                      executor=None, time_factor=1e6,
                      fitkind='linear', chunksize=10000,
                      overprovisioning_factor=0.01):
    """
    Parallel variant of serial_resample
    """
    new_t = resampled_timespace(t, new_samplerate, time_factor=time_factor)
    # Lazily compute the new timespan
    if out is None:
        out = np.zeros(len(new_t))
    if executor is None:
        executor = QueuedThreadExecutor()
    ovp_size = int(math.floor(overprovisioning_factor * chunksize))
    # How many chunks do we have to process?
    numchunks = len(new_t) // chunksize

    # Bind constant arguments
    f = functools.partial(__parallel_resample_worker, torig=t, tnew=new_t,
                          y=y, out=out, chunksize=chunksize,
                          ovp_size=ovp_size, prefilter=prefilter,
                          fitkind=fitkind)

    futures = [executor.submit(f, i=i) for i in range(numchunks)]
    # Wait for futures to finish
    concurrent.futures.wait(futures)

    return out

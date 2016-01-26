#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for selecting and finding specific attributes in datasets
"""
import numpy as np
import datetime
from bisect import bisect_left

__all__ = ["selectByDatetime", "selectFrequencyRange"]

def selectByDatetime(timestamps, time, factor=1.0, around=None, ofs=0.0):
    """
    Find the index in a timestamp array that is closest to a given value.
    The timestamp array expected to be sorted in ascending order. It is
    expected to contain Epoch timestamps as either integral or floating-point type.

    This function does perform any copying or modification of the array, so it is suitable
    for huge mmapped arrays.

    Depending on the scale of the timestamps, the factor argument needs to be adjusted
    accordingly. For example, 1e3 means that a difference of 1.0 in two timestamps means
    a difference of 1.0 ms. A factor of 1e6 means that a difference
    of 1.0 in two timestamp values means a difference of 1.0 µs. The default factor of 1.0
    means that 1.0 difference in two timestamps means a difference of 1.0 second and therefore
    is compatible with datetime.timestamp() values

    time may be either a datetime argument or a string in the %Y-%m-%d %H:%M:%S format.
    If the time string contains a decimal dot, the %Y-%m-%d %H:%M:%S.%f format, which
    allows specification up to microseconds, is used.

    Optionally, an offset value may be given that is substracted from the
    calculated timestamp before applying the factor.
    When using a timestamp offset of 1e9 with a factor of 1.0, for example,
    the value in the timestamp array for a given date is expected to be 1e9 less
    than the epoch time.

    This function uses bisect_left internally. Refer to its documentation
    for exact behavioural description.

    If around is None, returns the idx of the closest match.
    Else, returns (idx - around, idx + around), i.e. with idx right in the middle.
    This allows easy generation of ranges centered around a specific value.
    """
    if isinstance(time, str):
        if "." in time:
            time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
        else:  # No microsecond part
            time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    idx = bisect_left(timestamps, (time.timestamp() - ofs) * factor)
    if around is None:
        return idx
    else:  # Return range
        return (idx - around, idx + around)

def __computeFrequencyRangeIndices(x, lowFreq, highFreq):
    """
    Compute (startidx, endidx) for a given frequency array (e.g. fro FFT)
    """
    startidx = np.searchsorted(x >= lowFreq, True)
    endidx = np.searchsorted(x >= highFreq, True)
    return (startidx, endidx)

def selectFrequencyRange(x, y, lowFreq=1.0, highFreq=10.0):
    """
    From a FFT (x,y) pair, select only a certain frequency range. Returns (x,y)
    Use computeFrequencyRangeIndices() to get the indices.

    This function is designed to be inlined with a FFT call
    """
    startidx, endidx = __computeFrequencyRangeIndices(x, lowFreq, highFreq)
    # Remove everything except the selected frequency range
    return (x[startidx:endidx], y[startidx:endidx])

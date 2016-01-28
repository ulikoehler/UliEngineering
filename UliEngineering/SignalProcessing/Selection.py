#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for selecting and finding specific attributes in datasets
"""
import numpy as np
import datetime
import numbers
import scipy.signal
from bisect import bisect_left
from collections import namedtuple

__all__ = ["selectByDatetime", "selectFrequencyRange", "findSortedExtrema",
           "selectByThreshold", "findTrueRuns", "shrinkRanges", "IntInterval",
           "selectRandomSlice", "findNearestIdx"]

# Define interval class and override to obtain operator overridability
__Interval = namedtuple("Interval", ["start", "end"])


class IntInterval(__Interval):
    """
    Tuple-like type that represents an integral interval (or a slice) inside an
    integral space.

    This class:
        - allows easy ofsetting by e.g. adding a scalar and
    other convenience operations.
        - overrides __call__() for convenient slicing.
        - override __len__() for size determination
        - override __mul__ in a center-preserving manner

    Multiplying preserves the center of the interval (might be offset by one
        due to integral properties).
    Multiplying by 1.0 does not perform any change. Multiplication by 0.5
    halves the interval while multiplication by 4 quadruples its size.
    Multiplication doe not use 0-bounded arithmetic to allow multiply-then-add
    offsetting. If the multiplication factor is less than 1 but > 0,
    the interval will maintain a size of at least one.

    Multiplication by zero results in a zero-sized interval.
    """
    def __radd__(self, i):
        if not isinstance(i, numbers.Integral):
            raise ValueError("Can only add integers to an interval")
        return IntInterval(self.start + i, self.end + i)

    def __add__(self, i):
        return self.__radd__(i)

    def __rsub__(self, i):
        if not isinstance(i, numbers.Integral):
            raise ValueError("Can only substract integers from an interval")
        return IntInterval(i - self.start, i - self.end)

    def __sub__(self, i):
        if not isinstance(i, numbers.Integral):
            raise ValueError("Can only substract integers from an interval")
        return IntInterval(self.start - i, self.end - i)

    def __call__(self, arr):
        return arr[self.start:self.end]

    def __len__(self):
        return self.end - self.start

    def __mul__(self, n):
        if n == 1:
            return self
        elif n == 0:  # Return size-0 interval
            center = (self.end + self.start) // 2
            return IntInterval(center, center)
        elif n < 1:  # Shrink
            # Compute what to remove at each end
            toRemove = int(round(len(self) * (1.0 - n) / 2))
            return IntInterval(self.start + toRemove, self.end - toRemove)
        else:  # n > 1: Expand
            # Compute what to remove at each end
            toAdd = int(round(len(self) * (n - 1.0) / 2))
            return IntInterval(self.start - toAdd, self.end + toAdd)

    def __rmul__(self, n):
        return self.__mul__(n)

    def __truediv__(self, n):
        return self.__mul__(1.0 / n)

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
        return IntInterval(idx - around, idx + around)

def __computeFrequencyRangeIndices(x, lowFreq, highFreq):
    """
    Compute (startidx, endidx) for a given frequency array (e.g. fro FFT)
    """
    startidx = np.searchsorted(x >= lowFreq, True)
    endidx = np.searchsorted(x >= highFreq, True)
    return (startidx, endidx)

def selectFrequencyRange(x, y=None, lowFreq=1.0, highFreq=10.0):
    """
    From a FFT (x,y) pair, select only a certain frequency range. Returns (x,y)
    Use computeFrequencyRangeIndices() to get the indices.

    This function is designed to be inlined with a FFT call. In this case,
    x is a tuple (x, y)
    """
    if y is None:
        x, y = x
    startidx, endidx = __computeFrequencyRangeIndices(x, lowFreq, highFreq)
    # Remove everything except the selected frequency range
    return (x[startidx:endidx], y[startidx:endidx])

def __mapAndSortIndices(x, y, idxs, sort_descending=True):
    """
    Map a list of indices to a y array and sort it by y.
    This is used in multiple selectByX() functions.
    """
    xvals = x[idxs]
    yvals = y[idxs]
    idxs = np.argsort(yvals)
    if sort_descending:
        idxs = np.flipud(idxs)
    # Copy x/y values to new array
    return np.column_stack((xvals[idxs], yvals[idxs]))

def findSortedExtrema(x, y, comparator=np.greater, order=1, mode='clip'):
    """
    Find extrema using the given method and parameters, order them by y value and
    return a (n, 2)-shaped array that contains (for each extremum 0..n-1) the
    x and y value, with the 1st dimension being sorted in descending order.

    The comparator may be either np.greater or np.less.

    This means that ret[0] contains the x, y coordinate of the most significant extremum
    (where the significancy is determined by the comparator)
    """
    # Determine extrema and x/y values at those indices
    if comparator != np.greater and comparator != np.less:
        raise ValueError("Comparator may only be np.greater or np.less")
    extrema = scipy.signal.argrelextrema(y, comparator, 0, order, mode)[0]
    return __mapAndSortIndices(x, y, extrema, comparator == np.greater)

def selectByThreshold(fx, fy, thresh, comparator=np.greater):
    """
    Select values where a specific absolute threshold applies
    Returns a (n, 2)-shaped array where
    ret[i] = (x, y) contains the x and y values
    and the array is sorted in descending order by absolute y values.
    """
    if comparator != np.greater and comparator != np.less:
        raise ValueError("Comparator may only be np.greater or np.less")
    idxs = np.where(comparator(fy, thresh))
    return __mapAndSortIndices(fx, fy, idxs, comparator == np.greater)

def findTrueRuns(arr):
    """
    Find runs of True values in a boolean array.
    This function is not intended to be used with arrays other than Booleans.
    The end element of the ranges is inclusive.
    Return a (n, 2)-shaped array where the 2nd dimension contain start and end indices.
    """
    # Ensure the ends don't cause issues and we are calculating in int-space
    oneZero = np.zeros(1)
    cated = np.concatenate((oneZero, arr, oneZero)).view(dtype=np.int)
    diffs = np.diff(cated)
    starts = np.where(diffs >= 1)
    ends = np.where(diffs <= -1)
    return np.vstack((starts, ends)).T

__shrinkRangeMethodLUT = {
    "maxy": lambda start, _, yslice: start + np.argmax(yslice),
    "median": lambda start, end, _: (start + end) // 2
}

def shrinkRanges(ranges, y, method="maxy"):
    """
    Take a (n, 2)-shaped range list like the one returned by findTrueRuns()
    and shrink the ranges so the are only 1 wide.

    Currently supported shrinking methods are:
        - maxy: Selects the maximum y value along the slice
        - mean: Selects the index (start+end) // 2

    Return a 1d array of indices which are
    """
    ret = np.empty(ranges.shape[0])
    fn = __shrinkRangeMethodLUT[method]
    for i, (start, end) in enumerate(ranges):
        # Skip calculation for ranges which already are 1 wide
        if end - start == 1:
            ret[i] = start
        else:
            ret[i] = fn(start, end, y[start:end])
    return ret

def findNearestIdx(arr, v):
    """
    Find the index in the array which refers to the value with the least
    absolute difference from v.
    """
    # Original idea by unutbu @SO: http://stackoverflow.com/a/2566508/2597135
    return (np.abs(arr - v)).argmin()

def selectRandomSlice(arr, size):
    """
    Select a uniformly random slice of exactly a given size from the given array.

    Array may be a 1D numpy array or an integral which represents the array size.

    Return an IntInterval instance or raise if the array is not large enough
    """
    if isinstance(arr, numbers.Integral):
        alen = arr
    else:  # Assume numpy-like
        alen = arr.shape[0]
    if alen < size:
        msg = "Array of size {0} is not large enough to hold interval of size {1}"\
              .format(alen, size)
        raise ValueError(msg)
    elif alen == size:
        return IntInterval(0, alen)
    r = np.random.randint(0, alen - size)
    return IntInterval(r, r + size)

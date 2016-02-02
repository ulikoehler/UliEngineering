#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for selecting and finding specific attributes in datasets
"""
import numpy as np
import datetime
import numbers
import scipy.signal
from bisect import bisect_left, bisect_right
import collections

__all__ = ["selectByDatetime", "selectFrequencyRange", "findSortedExtrema",
           "selectByThreshold", "findTrueRuns", "shrinkRanges", "IntInterval",
           "selectRandomSlice", "findNearestIdx", "resample_discard",
           "GeneratorCounter", "majority_vote_all", "majority_vote",
           "extract_by_reference", "rangeArrayToIntIntervals",
           "intIntervalsToRangeArray", "applyRangesToArray"]

# Define interval class and override to obtain operator overridability
__Interval = collections.namedtuple("Interval", ["start", "end"])


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
            raise TypeError("Can only add integers to an interval")
        return IntInterval(self.start + i, self.end + i)

    def __add__(self, i):
        return self.__radd__(i)

    def __rsub__(self, i):
        if not isinstance(i, numbers.Integral):
            raise TypeError("Can only substract integers from an interval")
        return IntInterval(i - self.start, i - self.end)

    def __sub__(self, i):
        if not isinstance(i, numbers.Integral):
            raise TypeError("Can only substract integers from an interval")
        return IntInterval(self.start - i, self.end - i)

    def __call__(self, *args):
        if not args:
            raise TypeError("Use one or multiple arrays to produce a slice of them")
        if len(args) == 1:
            return args[0][self.start:self.end]
        return tuple([arr[self.start:self.end] for arr in args])

    def __len__(self):
        return self.end - self.start

    def __mul__(self, n):
        if not isinstance(n, numbers.Number):
            raise TypeError("Intervals can only be multiplied by numbers")
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
        if not isinstance(n, numbers.Number):
            raise TypeError("Intervals can only be divided by numbers")
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

def __computeFrequencyRangeIndices(x, low, high):
    """
    Compute (startidx, endidx) for a given frequency array (e.g. fro FFT)
    """
    startidx = np.searchsorted(x >= low, True) if low is not None else None
    endidx = np.searchsorted(x >= high, True) if high is not None else None
    return (startidx, endidx)

def selectFrequencyRange(x, y=None, low=None, high=None):
    """
    From a FFT (x,y) pair, select only a certain frequency range. Returns (x,y)
    Use computeFrequencyRangeIndices() to get the indices.

    This function is designed to be inlined with a FFT call. In this case,
    x is a tuple (x, y) and y is None (default).

    The low and/or high frequencies can be set to None to include the full
    frequency range, starting from or ending at a specific frequency.
    """
    if y is None:
        x, y = x
    startidx, endidx = __computeFrequencyRangeIndices(x, low, high)
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

def shrinkRanges(ranges, y=None, method="maxy"):
    """
    Take a (n, 2)-shaped range list like the one returned by findTrueRuns()
    and shrink the ranges so the are only 1 wide.

    Currently supported shrinking methods are:
        - maxy: Selects the maximum y value along the slice
        - mean: Selects the index (start+end) // 2. y may be None.

    Return a 1d array of indices which are
    """
    ret = np.empty(ranges.shape[0])
    fn = __shrinkRangeMethodLUT[method]
    needY = method != "median"
    if needY and y is None:
        raise TypeError("Y is required if method != median, but y is None")
    for i, (start, end) in enumerate(ranges):
        # Skip calculation for ranges which already are 1 wide
        if end - start == 1:
            ret[i] = start
        else:
            ret[i] = fn(start, end, y[start:end] if needY else  None)
    return ret


def applyRangesToArray(ranges, arr):
    """
    Apply a range array like the one returned by findTrueRuns().
    Yields each value
    :param ranges: The (n, 2) range array
    :param arr: The array to apply the ranges to
    """
    if ranges.shape[1] != 2:
        raise ValueError("ranges.shape[1] must be 2 instead of {0} - this does not look like a range array".format(ranges.shape[1]))
    for range in ranges:
        yield arr[range[0]:range[1]]


def rangeArrayToIntIntervals(ranges):
    """
    Convert a 2d range array, like the one returned by findTrueRuns(),
    to a list of int ranges).
    """
    return [IntInterval(r[0], r[1]) for r in ranges]


def intIntervalsToRangeArray(intervals):
    """
    Converts a list of IntIntervals to a 2d range array,
    like the one returned by findTrueRuns(),
    """
    return np.asarray([(interval[0], interval[1]) for interval in intervals])



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


class GeneratorCounter(object):
    """
    Utility class that provides zero-overhead counting for generators.
    At any point in time, len(...) of this class provides the number of
    items iterated so far

    Usage example:

        mycountinggen = CountingTee(mygen)
        myfunc(mycountinggen) # Same result as myfunc(mygen)
        print(len(mycountinggen))
    """
    def __init__(self, gen):
        self.gen = gen
        self.count = 0
        self.iter = self.gen.__iter__()

    def reiter(self, reset_count=False):
        """
        Attempts to restart iterating over the generator.
        This will work properly for lists, for example.

        Set reset_count to True to also reset the internal counter
        """
        if reset_count:
            self.count = 0
        self.iter = self.gen.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        result = next(self.iter)  # raises if no next value
        self.count += 1
        return result

    def __len__(self):
        return self.count


def majority_vote_all(arr, return_absolute=False):
    """
    Perform a majority selection on the value in an array.
    The values are required to be quantized, i.e. if no values are equal
    but only close together, this method won't work (try using a KDE-based meth0d)
    This algorithm is fast and works for huge datasets, however,
    and is well suited, for example, for analyzing FFT outputs
    (as FFT generates quantized outputs).

    majority_vote_all() return a list [(v, f)] which is ordered
    by descending commonness (equally common values are not ordered).
    f is the fraction of the total size of arr.
    If return_absolute == True, f is an absolute frequency.

    arr maybe any array-like structure or generator.
    """
    c = collections.Counter()
    gc = GeneratorCounter(arr)
    c.update(gc)
    most_common = c.most_common()
    if return_absolute:
        return c.most_common()
    return [(rec[0], rec[1] / len(gc)) for rec in c.most_common()]


def majority_vote(arr):
    """
    Wrapper for majority_vote_all() that only returns the most common value.
    The frequency of the value is ignored. None is returned if arr is empty.
    """
    mv = majority_vote_all(arr)
    return mv[0][0] if mv else None


def resample_discard(arr, divisor, ofs=0):
    """
    Resample with an integral divisor, discarding all other samples
    """
    return arr[ofs::divisor]


def extract_by_reference(fx, fy, ref):
    """
    When ref is an array of arbitrary timestamp-like values,
    extracts a range from fx, fy so that the range represented by
    ref matches the range represented by the return value as close as possible.

    Basically, when you have a time range, this function selects the same
    time range from another dataset, even if fx, fy and ref have different
    time resolution and time shifts.

    Returns (fx_new, fy_ne) so that fx_new[0] <= ref[0] and
    fx_new[-1] >= ref[-1] unless fx is too small.

    It is required that fx is sorted and ref[0] > ref[-1].
    A tuple (start, end) can be used instead of ref
    """
    start, end = ref[0], ref[-1]
    idx1 = bisect_left(fx, start)
    idx2 = bisect_left(fx, end)
    return fx[idx1:idx2], fy[idx1:idx2]

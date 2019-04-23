#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for selecting and finding specific attributes in datasets
"""
import numpy as np
import datetime
import numbers
import functools
from toolz import functoolz
from bisect import bisect_left, bisect_right
import collections

__all__ = ["select_by_datetime", "find_sorted_extrema",
           "select_by_threshold", "find_true_runs", "find_false_runs", "filter_runs",
           "runs_ignore_borders", "shrink_ranges", "IntInterval",
           "random_slice", "find_nearest_idx", "resample_discard",
           "GeneratorCounter", "majority_vote_all", "majority_vote",
           "extract_by_reference", "select_ranges",
           "sorted_range_indices", "multiselect", "find_closest_index"]

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
    @staticmethod
    def from_ranges(ranges):
        """
        Create a list of IntInterval instances from a range array

        Parameters
        ----------
        ranges : array_like
            A (n,2) array containing instances
        """
        return [IntInterval(r[0], r[1]) for r in ranges]

    @staticmethod
    def to_ranges(intervals):
        """
        Converts a list of IntInterval instances to a 2d range array,
        like the one returned by find_true_runs(),
        """
        return np.asarray([(interval[0], interval[1]) for interval in intervals])


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


def select_by_datetime(timestamps, time, factor=1.0, around=None, ofs=0.0, side="left"):
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
    bisect_func = {"left": bisect_left, "right": bisect_right}[side]
    idx = bisect_func(timestamps, (time.timestamp() - ofs) * factor)
    if around is None:
        return idx
    else:  # Return range
        return IntInterval(idx - around, idx + around)

def sorted_range_indices(arr, low, high):
    """
    Compute (startidx, endidx) for a given sorted array for a given low, high range
    so that all x in arr[startidx:endidx] is within (low, high)

    Commonly used for selecting frequency ranges from an FFT frequency.
    """
    startidx = np.searchsorted(arr >= low, True) if low is not None else None
    endidx = np.searchsorted(arr >= high, True) if high is not None else None
    return (startidx, endidx)

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

def _check_extrema_comparator(comparator):
    """
    Raise if the given comparator is neither np.greater nor np.less
    """
    if comparator != np.greater and comparator != np.less:
        raise ValueError("Comparator may only be np.greater or np.less")


try:
    import scipy.signal
    def find_sorted_extrema(x, y, comparator=np.greater, order=1, mode='clip'):
        """
        Find extrema using the given method and parameters, order them by y value and
        return a (n, 2)-shaped array that contains (for each extremum 0..n-1) the
        x and y value, with the 1st dimension being sorted in descending order.

        The comparator may be 

        This means that ret[0] contains the x, y coordinate of the most significant extremum
        (where the significancy is determined by the comparator)

        Parameters
        ----------
        mode : string
            How the edges of the vector are treated.
            Either 'clip', 'raise' or 'wrap',
            see numpy.take for more details
        comparator:
            Either np.greater or np.less.
            np.greater => Find maxima
            np.less => Find minima
        """
        _check_extrema_comparator(comparator)
        # Determine extrema and x/y values at those indices
        extrema = scipy.signal.argrelextrema(y, comparator, 0, order, mode)[0]
        return __mapAndSortIndices(x, y, extrema, comparator == np.greater)
except ModuleNotFoundError:
    def find_sorted_extrema(*args, **kwargs):
        raise NotImplementedError("You need to install scipy to use find_sorted_extrema()!")

def select_by_threshold(fx, fy, thresh, comparator=np.greater):
    """
    Select values where a specific absolute threshold applies
    Returns a (n, 2)-shaped array where
    ret[i] = (x, y) contains the x and y values
    and the array is sorted in descending order by absolute y values.
    """
    _check_extrema_comparator(comparator)
    idxs = np.where(comparator(fy, thresh))
    return __mapAndSortIndices(fx, fy, idxs, comparator == np.greater)

def find_true_runs(arr):
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
    starts = np.nonzero(diffs >= 1)[0]
    ends = np.nonzero(diffs <= -1)[0] - 1
    return np.vstack((starts, ends)).T

def find_false_runs(arr):
    """Alias for find_true_runs(np.logical_not(arr))"""
    return find_true_runs(np.logical_not(arr))

def runs_ignore_borders(runs, size=-1, ignore_start=True, ignore_end=True):
    """
    Ignore the first and/or the last run if they start at 0 or end at the array size respectively.
    
    Parameters
    ----------
    runs : array_like
        A (n,2) array such as returned by find_true_runs()
    size : int
        The size of the original array. Must be set appropriately when the end range shall be removed
    ignore_start : bool
        If true, the first run will be ignored if its start index is 0
    ignore_end : bool
        If true, the first run will be ignored if its start index is 0.
        The size parameter must be set correctly so it can be recognized
        if the last run ends at the array end.
        If size=-1, ignore_end is forced to False
    """
    startidx = 1 if (runs.size and ignore_start and runs[0,0] == 0) else None
    endidx = -1 if (runs.size and ignore_end and runs[-1,1] == size) else None
    return runs[startidx:endidx]


def __run_size_filter(minsize, maxsize):
    """Return a run filter function that checks >= minsize && <= maxsize."""
    def _filt(run):
        delta = run[1] - run[0] # It is assumed it's a correct run, i.e. run.size == 2 && b > a
        return delta >= minsize and delta <= maxsize
    return _filt

def filter_runs(runs, minsize=2, maxsize=np.inf):
    """
    Given a (n,2) array such as returned by findTrueRuns(), returns a new
    (n-x,2) run list that contains only runs at least minSize in size and
    """
    return np.asarray(list(filter(__run_size_filter(minsize, maxsize), runs)))

def __select_y(ranges, y, selector):
    """maxy selector for shrink_ranges"""
    return np.asarray([start + selector(y[start:end + 1]) for start, end in ranges])

__shrinkRangeMethodLUT = {
    "min": lambda arr: arr[:, 0],
    "max": lambda arr: arr[:, 1],
    "middle": lambda arr: (arr[:, 0] + arr[:, 1]) // 2,
    # Special
    "miny": functools.partial(__select_y, selector=np.argmin),
    "maxy": functools.partial(__select_y, selector=np.argmax),
}

def shrink_ranges(ranges, method="middle", **kwargs):
    """
    Take a (n, 2)-shaped range list like the one returned by find_true_runs()
    and shrink the ranges so they are only 1 wide. Returns a (n)-shaped array of indices.

    Currently supported shrinking methods are:
        - min: Selects the first index of the slice
        - max: Selects the last index of the slice
        - middle: Selects the index (start+end) / 2. y may be None.

    There also some special reducers that require additional kwargs
        - maxy Requires y=array_like kwarg. Selects the index where y is maximal in the range.
        - miny Requires y=array_like kwarg. Selects the index where y is minimal in the range.

    """
    return __shrinkRangeMethodLUT[method](ranges, **kwargs)


def select_ranges(ranges, arr):
    """
    Apply a range array like the one returned by find_true_runs().
    Yields each array slice in order

    Parameters
    ----------
    ranges : array_like
        The (n, 2) range array
    arr : array_like
        The array to apply the ranges to
    """
    for range in ranges:
        yield arr[range[0]:range[1]]



def ranges_to_IntInterva(ranges):
    """
    Convert a 2d range array, like the one returned by find_true_runs(),
    to a list of int ranges).
    """


def find_nearest_idx(arr, v):
    """
    Find the index in the array which refers to the value with the least
    absolute difference from v.
    """
    # Original idea by unutbu @SO: http://stackoverflow.com/a/2566508/2597135
    return (np.abs(arr - v)).argmin()


def random_slice(arr, size):
    """
    Select a uniformly random slice of exactly a given size from the given array.

    Array may be a 1D numpy array or an integral which represents the array size.

    Returns an IntInterval instance or raise if the array is not large enough
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
    but only close together, this method won't work (try using a KDE-based method)
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
        return most_common
    return [(rec[0], rec[1] / len(gc)) for rec in most_common]


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


def multiselect(lst, indices, convert=functoolz.identity):
    """
    Creates a new list from a list-like object, selecting only the indices
    in the index list, in the specified order.

    This works like numpy indexing with an index array.

    Parameters
    ----------
    lst : object with [] operator
        An object where all indices as listed in indices
        can be used
    indices : iterable
        The indices to use for selecting from lst.
    convert : unary function
        This function is called on indices[i] for every iteration.
        This can be e.g. set to tuple if lst accepts tuple indices
        but the indices argument consists of lists.

    Examples
    --------
    >>> multiselect([1,2,3,4,5,6], [3,1,5])
    [4, 2, 6]
    """
    return [lst[convert(idx)] for idx in indices]


def find_closest_index(frequencies, frequency):
    """
    Find the closest frequency bin in an array of frequencies
    and return its index in the frequency array.
    """
    return np.argmin(np.abs(frequencies - frequency))

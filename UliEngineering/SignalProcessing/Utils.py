#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsorted signal processing utilities
"""
import numpy as np
from toolz import functoolz
import numbers
import warnings
from .Selection import find_true_runs
from UliEngineering.EngineerIO import normalize_numeric

__all__ = ["remove_mean", "rms", "peak_to_peak", "unstair", "optimum_polyfit", "LinRange", "aggregate", "zero_crossings", "rms_to_peak_to_peak"]

_unstep_reduction_methods = {
    "left": lambda a: a[:, 0],
    "right": lambda a: a[:, 1],
    "middle": lambda a: np.sum(a, axis=1) // 2,
    "reduce": lambda a: a.flatten()
}

def remove_mean(arr):
    """
    Substract the DC signal component, i.e. the arithmetic mean of the array,
    from the array and return the modified array
    """
    return arr - np.mean(arr)

def rms(arr):
    """
    Compute the root-mean-square value of the given array
    """
    return np.sqrt(np.mean(np.square(arr)))

def rms_to_peak_to_peak(rms_val):
    """
    Given an RMS value, returns the peak to peak value
    of a sinusoid signal with that RMS value.
    """
    rms_val = normalize_numeric(rms_val)
    return rms_val * np.sqrt(2)

def peak_to_peak(arr):
    """
    Compute max(arr) - min(arr)
    """
    if arr is None or len(arr) == 0:
        # This causes numpy ValueError since some Numpy version
        return 0.
    return np.max(arr) - np.min(arr)

def unstair(x, y, method="diff", tolerance=1e-9):
    """
    Remove stairs (adjacent equal values) from a function with quantized values.
    The first and the last values are always returned.

    Currently available methods:
        - left: Use the leftmost value of every step
        - middle: Use shrinked ranges to return values in the middle of a step range
        - right: Use the rightmost value of every step
        - reduce: Return exactly the same function, but remove intermediate values below the tolerance.
                  This can be used for data reduction without actually changing the value.

    :param tolerance: The diff value allowed to assume a step.
                      The diff is compared to this on a per-sample basis.
    :return A tuple (x, y) with the corresponding new values.
    """
    # Separate steps and values which are not in a step
    stairs = np.abs(np.diff(y)) < tolerance
    normals = np.logical_not(stairs) # Values which are not in a step
    # Add first and last index. np.diff() returns idxs backshifted by one, so add 1 now
    normal_idxs = np.concatenate(([0, y.size - 1], normals.nonzero()[0] + 1))
    # Convert step sequences to ranges and remove some of their samples, depending on the method
    zero_runs = find_true_runs(stairs)
    zero_runs[:,1] += 1 # End index must not be inclusive
    stair_idxs = _unstep_reduction_methods[method](zero_runs)
    # Return all normal values, plus the reduced step values
    idxs = np.concatenate((normal_idxs, stair_idxs))
    # Postprocessing for special methods
    if method in ["right", "middle"]:
        idxs = np.setdiff1d(idxs, _unstep_reduction_methods["left"](zero_runs))
        # Add first index if not present
        if not (idxs == 0).any():
            idxs = np.concatenate(([0], idxs))
        if not (idxs == 0).any():
            idxs = np.concatenate((idxs, [y.size - 1]))
    else:
        idxs = np.unique(idxs)
    return x[idxs], y[idxs]


def optimum_polyfit(x, y, score=functoolz.compose(np.max, np.abs), max_degree=50, stop_at=1e-10):
    """
    Optimize the degree of a polyfit polynomial so that score(y - poly(x)) is minimized.

    :param max_degree: The maximum degree to try. LinAlgErrors are automatically ignored.
    :param stop_at: If a score lower than this is reached, the function returns early
    :param score: The score function that is applied to y - poly(x). Default: max deviation.
    :return A tuple (poly1d object, degree, score)
    """
    scores = np.empty(max_degree - 1, dtype=np.float64)
    # Ignore rank warnings now, but do not ignore for the final polynomial if not early returning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        for deg in range(1, max_degree):
            # Set score to max float value
            try:
                poly = np.poly1d(np.polyfit(x, y, deg))
            except np.linalg.LinAlgError:
                scores[deg - 1] = np.finfo(np.float64).max
                continue
            scores[deg - 1] = score(y - poly(x))
            # Early return if we found a polynomial that is good enough
            if scores[deg - 1] <= stop_at:
                return poly, deg, scores[deg - 1]
    # Find minimum score
    deg = np.argmin(scores) + 1
    # Compute polyfit for that degreet
    poly = np.poly1d(np.polyfit(x, y, deg))
    return poly, deg, np.min(scores)


class LinRange(object):
    """
    Combines the properties of numpy.linspace and Python3's range by providing
    a floating-point capable lazy range generator that does not keep the entire array
    in memory (but calculates slices on the fly.

    Use [:] or any other slice to obtain a numpy array. dtype is used as a wrapper function.
    Requires the use of numpy dtypes.

    Behaves like np.linspace. Use .view() to obtain a LinRange slice.
    """
    def __init__(self, start, stop, n, endpoint=True, dtype=float):
        "Create a new LinRange object using a numpy.linspace-like constructor"
        self.start = start
        self.stop = stop
        n = int(n)
        self.endpoint = endpoint
        self.step = (stop - start) / (n - 1 if endpoint else n)
        self.size = n
        self.dtype = dtype

    @staticmethod
    def range(start, stop, step):
        "Create a new LinRange object using a range()-like constructor"
        return LinRange(start, stop, int((stop - start) / step))

    def __len__(self):
        return self.size

    @property
    def mid(self):
        "Return the middle of the current interval as a floating point value"
        return self.dtype((self.start + self.stop) / 2.)

    @property
    def shape(self):
        return (self.size,)

    def astype(self, typearg):
        """
        Return self copy with a different data type
        """
        return LinRange(self.start, self.stop, self.size, dtype=typearg)

    def copy(self):
        """
        Return self as a numpy array.
        Designed to be compatible with numpy objects
        """
        return np.linspace(self.start, self.stop, self.size,
                           endpoint=self.endpoint, dtype=self.dtype)

    def view(self, start=None, stop=None, step=None):
        """
        Return a LinSpace view of self
        """
        istart, istop, istep = slice(start, stop, step).indices(self.size)
        return LinRange(self[istart], self[istop], (istop - istart) // istep, endpoint=False)

    def __getitem__(self, key):
        """
        Get:
            - A numpy linrange slice
        """
        if isinstance(key, slice):
            istart, istop, istep = key.indices(self.size)
            return np.linspace(self[istart], self[istop], (istop - istart) // istep,
                               endpoint=False, dtype=self.dtype)
        elif isinstance(key, numbers.Number):
            if key < 0:
                key = len(self) + key  # NOTE: Key is negative, so result is < len(self)!
            val = self.start + self.step * key
            # Convert to dtype
            if self.dtype == float:
                return val
            else:
                # TODO find better method, maybe using a scalar
                return np.asarray([val]).astype(self.dtype)[0]
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(type(key)))

    def __dtype_name(self):
        if hasattr(self.dtype, "__qualname__"):
            return self.dtype.__qualname__
        else:
            return str(self.dtype)

    def __repr__(self):
        return "LinRange({}, {}, {}{})".format(
            self.start,
            self.stop,
            str(self.step) if type(self.step) == np.timedelta64 else self.step,
            "" if self.dtype == float else ", dtype={}".format(self.__dtype_name())
            )

    def __eq__(self, other):
        return self.start == other.start and self.stop == other.stop and self.step == other.step

    def samplerate(self):
        """
        Returns the samplerate as float (unit: Hz).
        This works especially if step is a np.timedelta64 object.
        Else "step" is assumed to be a <seconds> value
        """
        if type(self.step) == np.timedelta64:
            ns = self.step.astype("timedelta64[ns]").astype(int)
            s = ns*1e-9
            return 1./s
        else:
            return 1./self.step

def aggregate(gen):
    """
    Takes any iterable and aggregates subsequent values
    yielded by the iterable into a single value with a counter.

    Yields (value, count) pairs

    Will NOT handle None values correctly
    """
    current = None
    cnt = 0
    for item in gen:
        if item == current:
            cnt += 1
        else:
            if current is not None:
                yield (current, cnt)
            current = item
            cnt = 1
    # Yield remaining item, if any
    if current is not None:
        yield (current, cnt)

def zero_crossings(data):
    """
    Compute indexes in the given data array just before a zero crossing occurs.
    A zero crossing at index i is defined as:
        - data[i] is positive
        - data[i + 1] exists and is negative
    or:
        - data[i] is negative
        - data[i + 1] exists and is positive.

    Returns an 1D numpy array with indexes.

    Thanks to Jim Brissom on Stack Overflow for this solution:
    https://stackoverflow.com/a/3843124/2597135
    """
    return np.where(np.diff(np.sign(data)))[0]

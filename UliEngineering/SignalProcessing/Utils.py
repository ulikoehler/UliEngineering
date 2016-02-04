#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsorted signal processing utilities
"""
import numpy as np
from toolz import functoolz
import numbers
import math
import warnings
from .Selection import shrinkRanges, findTrueRuns

__all__ = ["unstair", "optimum_polyfit", "LinRange"]


_unstep_reduction_methods = {
    "left": lambda a: a[:,0],
    "right": lambda a: a[:,1],
    "middle": lambda a: np.sum(a, axis=1) // 2,
    "reduce": lambda a: a.flatten()
}

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
    zero_runs = findTrueRuns(stairs)
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
    in memory (but calculates slices on the fly
    """
    def __init__(self, start, stop, n, endpoint=True):
        "Create a new LinRange object using a numpy.linspace-like constructor"
        self.start = start
        self.stop = stop
        n = int(n)
        self.step = (stop - start) / (n - 1 if endpoint else n)
        self.size = n

    @staticmethod
    def range(start, stop, step):
        "Create a new LinRange object using a range()-like constructor"
        return LinRange(start, stop, int((stop - start) / step))

    def __len__(self):
        return self.size

    def view(self, start, stop, step=1):
        """Return a slice of this LinRange as a view, not as a numpy array"""
        istart, istop, istep = slice(start, stop, step).indices(self.size)
        return LinRange(self[istart],
                        self[istop - 1],
                        (istop - istart) / istep)

    @property
    def mid(self):
        "Return the middle of the current interval as a floating point value"
        return (self.start + self.stop) / 2.

    def __getitem__(self, key):
        if isinstance(key, slice):
            istart, istop, istep = key.indices(self.size)
            start = self[istart]
            stop = self[istop - 1]
            n = (istop - istart) / istep
            return np.linspace(start, stop, n)
        elif isinstance(key, numbers.Number):
            if key < 0:
                key = len(self) + key
            return self.start + self.step * key
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(type(key)))

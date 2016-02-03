#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsorted signal processing utilities
"""
import numpy as np
from .Selection import shrinkRanges, findTrueRuns

__all__ = ["unstair"]


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
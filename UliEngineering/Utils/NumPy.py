#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re
from datetime import datetime

__all__ = ["numpy_resize_insert", "invert_bijection", "apply_pairwise_1d",
           "ngrams", "split_by_pivot", "datetime64_now",
           "timedelta64_resolution", "datetime64_resolution"]

def numpy_resize_insert(arr, val, index, growth_factor=1.5, min_growth=1000, max_growth=1000000):
    """
    Append a value to a 1D numpy array. Resize dynamically if required.
    Returns the new array (which may be the same as the old array).
    No resize is performed if the array is already large enough

    Supports non-continous insertion of values (i.e. index must not grow monotonically).

    After finishing to call numpy_resize_insert, the user must call
    np.resize(arr, actual_size) to trim extra elements from the array.

    The new size is calculate as
    max(min(max(factor_growth, min_growth), max_growth), required_growth)

    Parameters
    ----------
    arr : array_like
        The original array
    val
        The new value to insert
    index : int > 0
        The index to insert value at
    growth_factor : float
        What fraction of the current size the array will be resized to
        in case of a resize
    min_growth : int
        The minimum array growth expressed as number of elements.
        Setting this higher might slightly increase memory overhead,
        but drastically decreases reallocation frequency and therefore significantly increases
        speed.
    max_growth : int
        An upper bound on the extra array size. Setting this too high increases memory overhead,
        but setting it too low will cause too small allocations to occur, resulting in more
        frequent reallocations and therefore more running speed.
    """
    # Array large enough ? If so, we don't have to bother about insertion.
    if index < arr.size:
        arr[index] = val
        return arr
    # OK, need to resize
    required_growth = (index - arr.size) + 1
    factor_growth = int(arr.size * growth_factor)
    # Resize at least min_growth & at most max_growth, but at least required_growth (overrides all other rules)
    growth = max(min(max(factor_growth, min_growth), max_growth), required_growth)
    # Resize (empty space is filled with copies of original array)
    arr = np.resize(arr, arr.size + growth)
    arr[index] = val
    return arr


def invert_bijection(arr):
    """
    Performs an inversion on a bijective array such as arrays
    returned by np.argsort().

    For each value x in the given array at index i,
    the new array will

    Preconditions (not checked):
        - All elements must be >= 0 and < arr.size
        - Array must contain each element < arr.size exactly once
        - Array values must be usable as indices

    Parameters
    ----------
    arr : iterable
        The source array to invert. Must meet the listed preconditions.

    Examples
    --------
    >>> invert_bijection(np.arange(4))
    array([0, 1, 2, 3])
    >>> invert_bijection(np.asarray([1,2,0,3]))
    array([2,0,1,3])
    >>> invert_bijection(np.asarray([1,0,2,3]))
    array([1,0,2,3])
    >>> invert_bijection([1,0,2,3])
    array([1,0,2,3])
    """
    ret = np.zeros_like(arr)
    for new, old in enumerate(arr):
        ret[old] = new
    return ret


def apply_pairwise_1d(valuesA, valuesB, fn, dtype=float):
    """
    Given two 1d arrays, generates a 2d matrix
    containing at any coordinate [x,y] the value of fn(valuesA[x], valuesB[y]).
    If valu

    The input values do not neccessarily have to be numbers and can be
    non-uniform throughout the input list data type,
    however the output (and hence the return type of fn)
    must be uniform and supported by numpy. The exact data type
    must be specified in the dtype parameter

    Parameters
    ----------
    valuesA : array-like
        The first array of input values to fn.
    valuesB : array-like
        The second array of input values to fn.
    fn : binary function
        Applied to input value pairs
    dtype : numpy-supported data type
        The datatype that is passed when generating the result matrix.
    """
    valuesB = valuesA if valuesB is None else valuesB
    # TODO: Possibly there is a better algorithm, e.g. np.apply_along_axis?
    n, m = len(valuesA), len(valuesB)
    result = np.zeros((n, m), dtype=dtype)
    for x in range(n):
        for y in range(m):
            result[x, y] = fn(valuesA[x], valuesB[y])
    return result


def ngrams(arr, n, closed=False):
    """
    Yield ngrams of subsequent entries from an arbitrarily-shaped array.
    For example, with arr=[1,2,3,4,5,6]
    and n=2, yields [[1,2][2,3],[3,4],[4,5],[5,6]]
    if closed=False or [[1,2][2,3],[3,4],[4,5],[5,6],[6,1]] if closed=True.

    For n=2, this function behaves similarly to
    UliEngineering.Math.Geometry.polygon_lines()
    however this function supports arbitrarily-shaped arrays.

    Parameters
    ----------
    arr : numpy array like
        The source array.
        Will not be modified
    n : int
        The length of each ngram.
        Must be <= array.shape[0].
    closed : bool
        True if there should be an ngram containing the last
        and the first entries (until the next ngram would be
        the first in the list)
    """
    idxs = np.arange(n)
    for i in range(arr.shape[0] if closed else arr.shape[0] - n + 1):
        yield arr[idxs]
        idxs = np.mod(idxs + 1, arr.shape[0])


def split_by_pivot(arr, pivots):
    """
    Takes a numpy array and splits it according to pivot points.
    Yields each slice.
    Examples:
    split_by_pivot([0,1,2,3,4,5], [2]) => [[0,1],[2,3,4,5]]
    split_by_pivot([0,1,2,3,4,5], [2,4]) => [[0,1],[2,3],[4,5]]

    Parameters
    ----------
    arr : any sliceable iterable
        The array to return slice froms
    pivots : iterable of ints
        The pivot points to split at
    """
    # Get all slice endpoints
    # list() required for numpy arrays
    idxs = [0] + list(pivots) + [len(arr)]
    # Generate all slice endpoint pairs
    for start, end in zip(idxs[:-1], idxs[1:]):
        yield arr[start:end]

def datetime64_now():
    """
    Return datetime.now() as np.datetime64 object.
    """
    return np.datetime64(datetime.now())

# Regex for timedelta64_resolution etc
_resolution_re = re.compile(r'^[^\[]+\[([^\]]+)\]$')

def timedelta64_resolution(tdelta):
    """
    Given a timedelta64 object, returns its resolution as a string,
    e.g. 'us' or 'ms'.
    """
    s = str(tdelta.dtype) # e.g. 'timedelta64[us]'
    match = _resolution_re.match(s)
    if match is None:
        raise ValueError("Data type {} is not supported for ..._resolution!".format(s))
    else:
        return match.group(1)

def datetime64_resolution(dt):
    """
    Given a datetime64 object, returns its resolution as a string,
    e.g. 'us' or 'ms'.
    """
    return timedelta64_resolution(dt) # Same algorithm!

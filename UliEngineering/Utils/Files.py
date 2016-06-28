#!/usr/bin/env python3
"""
File utilities
"""
from toolz import functoolz
import operator
import numpy as np

__standard_isline = functoolz.compose(bool, str.strip)
# Utility to get nth CSV column
_csv_nthcol = lambda n: functoolz.compose(operator.itemgetter(n), lambda s: s.partition(','))
_csv_firstcol = _csv_nthcol(0)

def count_lines(flo, isline=__standard_isline):
    """
    Count the lines in a file.

    Takes a file-like object. Strings are treated as filenames.
    Returns the number of lines.
    """
    # Open it if it is a string
    if isinstance(flo, str):
        with open(flo, "r") as infile:
            return count_lines(infile, isline=isline)
    # Actual counting code
    num_lines = 0
    for line in flo:
        num_lines += 1 if isline(line) else 0
    return num_lines


def extract_numeric_column(flo, isline=__standard_isline, postproc=functoolz.identity,
                           extractcol=_csv_firstcol, initsize=10000, **kwargs):
    """
    Like extract_column, but places the results in a numpy array
    """
    # Open it if it is a string
    if isinstance(flo, str):
        with open(flo, "r") as infile:
            return extract_numeric_column(infile, isline=isline, postproc=postproc, extractcol=extractcol,
                                          initsize=initsize, **kwargs)
    # Actual counting code
    index = 0 # Current 0-based index in array
    arr = np.zeros(initsize)
    for line in flo:
        if not isline(line): continue
        val = postproc(extractcol(line))
        arr = numpy_resize_insert(arr, val, index, **kwargs)
        index += 1
    # Trim to size. Index is now [last written index ] + 1 which is the size of the array
    np.resize(arr, index)
    return arr

def numpy_resize_insert(arr, val, index, growth_factor=1.5, min_growth=1000, max_growth=1000000):
    """
    Append a value to a 1D numpy array. Resize dynamically if required.
    Returns the new array (which may be the same as the old array)
    """
    # Array large enough ? If so, we can resize directly
    if index < arr.size:
       arr[index] = val
       return arr
    # OK, need to resize
    required_growth = (index - arr.size) + 1
    factor_growth = int(arr.size * growth_factor)
    # Resize at least min_growth & at most max_growth, but at least required_growth (overrides all other rules)
    growth = max(min(max(factor_growth, min_growth), max_growth), required_growth)
    #print("Resizing from {0} to {1}".format(arr.size, arr.size + growth))
    # Resize (empty space is filled with copies of original array)
    arr = np.resize(arr, arr.size + growth)
    arr[index] = val
    return arr


def extract_column(flo, isline=__standard_isline, postproc=functoolz.identity,
                   extractcol=_csv_firstcol):
    """
    Lazily extract a column from a file, for example extract a column from a CSV file.
    The values are run through a postprocessing function and placed in a list which is returned.
    Lines which do not pass the isline function are ignored.
    """
    # Open it if it is a string
    if isinstance(flo, str):
       with open(flo, "r") as infile:
         return extract_column(infile, isline=isline, postproc=postproc)
    # Actual counting code #TODO
    columns = []
    for line in flo:
        if not isline(line): continue
        line = preproc(line)
        col = postproc(extractcol(line))
        columns.append(col)
    return columns

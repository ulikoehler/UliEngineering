#!/usr/bin/env python3
"""
File utilities
"""
from toolz import functoolz
import operator
import numpy as np
from .NumPy import numpy_resize_insert

_strip_newline = lambda s: s.strip("\n")
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
                           preproc=_strip_newline, extractcol=_csv_firstcol, initsize=10000, **kwargs):
    """
    Like extract_column, but places the results in a numpy array
    """
    # Open it if it is a string
    if isinstance(flo, str):
        with open(flo, "r") as infile:
            return extract_numeric_column(infile, isline=isline, postproc=postproc,
                                          preproc=preproc, extractcol=extractcol,
                                          initsize=initsize, **kwargs)
    # Actual counting code
    index = 0  # Current 0-based index in array
    arr = np.zeros(initsize)
    for line in flo:
        if not isline(line): continue
        line = preproc(line)
        val = postproc(extractcol(line))
        arr = numpy_resize_insert(arr, val, index, **kwargs)
        index += 1
    # Trim to size. Index is now [last written index ] + 1 which is the size of the array
    return np.resize(arr, index)


def extract_column(flo, isline=__standard_isline, preproc=_strip_newline,
                   postproc=functoolz.identity, extractcol=_csv_firstcol):
    """
    Lazily extract a column from a file, for example extract a column from a CSV file.
    The values are run through a postprocessing function and placed in a list which is returned.
    Lines which do not pass the isline function are ignored.

    The postprocessing function may return None, in which case the line is ignored.
    """
    # Open it if it is a string
    if isinstance(flo, str):
        with open(flo, "r") as infile:
            return extract_column(infile, isline=isline, postproc=postproc, preproc=preproc, extractcol=extractcol)
    # Actual counting code #TODO
    columns = []
    for line in flo:
        if not isline(line): continue
        line = preproc(line)
        col = postproc(extractcol(line))
        if col is not None:
            columns.append(col)
    return columns

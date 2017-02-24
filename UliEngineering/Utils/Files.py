#!/usr/bin/env python3
"""
File utilities
"""
from toolz import functoolz
import operator
import os
import os.path
import numpy as np
from .NumPy import numpy_resize_insert

__all__ = [
    "count_lines",
    "extract_numeric_column",
    "extract_column",
    "write_textfile",
    "read_textfile",
    "list_recursive",
    "find_datasets_by_extension"
]

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

def write_textfile(path, text):
    """
    Utility to write text to a file,
    auto-creating the directory tree
    Does not write a terminating newline.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as outfile:
        outfile.write(text)

def read_textfile(path):
    """
    Utility to read utf-8 encoded text from a file
    """
    with open(path, "r") as infile:
        return infile.read()

def list_recursive(directory, relative=False, files_only=True):
    """
    List a directory recursively, yielding each filename
    (with the filename being relative to the directory).
    
    The results are generated in no particular order.

    Parameters
    ----------
    directory : str | path-like
        The directory to list
    relative : bool
        If True, yield relative paths
    files_only : bool
        If True, yield only files and ignore directories.s
        If False, yield directories (the name ends with a slash)
        The root directory is never yielded
    """
    for dirname, subdirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(dirname, file)
            yield os.path.relpath(path, directory) if relative else path
        if not files_only:
            for subdir in subdirs:
                path = os.path.join(dirname, subdir) + "/"
                yield os.path.relpath(path, directory) + "/" if relative else path

def find_datasets_by_extension(namelist, extensions):
    """
    Searches for datasets that are located in the same directory
    and have the same name, but different extensions.
    
    Finds only datasets where all extensions are present
    within a single directory with a single basename.
    
    Usually you would acquire the namelist using
    list(UliEngineering.Utils.Files.list_recursive()) or
    UliEngineering.Utils.ZIP.list_zip()
    
    Example:
    find_datasets_by_extension(list_zip("my.zip"), (".dbf", ".prj", ".shp"))
    
    Parameters
    ----------
    namelist : iterable of strings
        A recursive directory list to find the datasets in
    extensions : iterable of strings (usually tuple)
        A list or tuple of extensions, starting with a dot,
        of files in the dataset that need to be present
        with the same prefix
    
    Yields
    ------
    A list of filenames (each in the namelist) for
    each dataset found
    """
    if len(extensions) == 0:
        raise ValueError("At least one extension must be given")
    if not all(ext.startswith(".") for ext in extensions):
        raise ValueError("All extensions must start with a dot!")
    # We start by finding all files with the first extension
    index_ext = extensions[0]
    index_prefixes = [os.path.splitext(name)[0] # Prefix
                   for name in namelist
                   if os.path.splitext(name)[1] == index_ext]
    # Find 
    for prefix in index_prefixes:
        extfiles = [prefix + ext for ext in extensions]
        if all(extfile in namelist for extfile in extfiles):
            yield(extfiles)

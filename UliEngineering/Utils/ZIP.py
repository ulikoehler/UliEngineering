#!/usr/bin/env python3
"""
Utilities for ZIP files
"""
import io
import os.path
import zipfile
from .Files import list_recursive

__all__ = ["create_zip_from_directory", "list_zip", "read_from_zip"]

def create_zip_from_directory(zippath, directory, include_rootdir=True):
    """
    Create a ZIP file from a directory that exist
    on the filesystem. Adds all files recursively,
    naming them correctly.

    Parameters
    ----------
    zippath : path-like
        The path of the ZIP file to write
    directory : path-like
        The directory to compress
    include_rootdir : bool
        if True, the basename of the directory is prepended
        to each filename in the ZIP (i.e. when running unzip
        on the ZIP, one directory is extracted)
    """
    basename = os.path.basename(directory)
    with zipfile.ZipFile(zippath, mode="w") as zipout:
        # Find files in directory
        for filename in list_recursive(directory, relative=True):
            filepath = os.path.join(directory, filename)
            # Write with custom name
            zipout.write(filepath,
                         os.path.join(basename, filename)
                         if include_rootdir else filename)

def list_zip(zippath):
    """
    Get a list of entries in the ZIP.
    Equivalent to calling .namelist() on the
    opened ZIP file.
    """
    with zipfile.ZipFile(zippath) as zipin:
        return zipin.namelist()

def read_from_zip(zippath, filepaths, binary=True):
    """
    Read one or multiple files from a ZIP, copying their contents to memory.
    
    Parameters
    ----------
    zippath : path-like
        The path of the ZIP file
    filepath : str or iterable of strings
        The path of the file inside the ZIP
        Multiple paths allowed (=> list is returned)
    binary : bool
        If True, returns a io.BytesIO().
        If False, returns a io.StringIO()
        
    Returns
    -------
    If filepath is a string, a single file-like object (in-memory).
    If filepath is any other iterable, a list of file-like in-memory objs.
    """
    iof = io.BytesIO if binary else io.StringIO
    # Handle single file using the same code as multiple files
    single_file = isinstance(filepaths, str)
    filepaths = [filepaths] if single_file else filepaths
    # Actually
    with zipfile.ZipFile(zippath) as thezip:
        # Read multiple files
        iobufs = []
        for file in filepaths:
            with thezip.open(file) as inf:
                iobufs.append(iof(inf.read()))
        # Return result
        return iobufs[0] if single_file else iobufs
            
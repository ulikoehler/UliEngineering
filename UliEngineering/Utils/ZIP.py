#!/usr/bin/env python3
"""
Utilities for ZIP files
"""
import io
import os.path
import zipfile
from .Files import list_recursive

__all__ = ["create_zip_from_directory", "list_zip"]

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities regarding temporary storage."""
import tempfile
import shutil
import os

class AutoDeleteTempfileGenerator:
    
    """A wrapper for temporary files and directories that are automatically deleted.

    Once this class is deleted or delete_all() is called, the files are deleted.

    This is not comparable to tempfile.TemporaryFile as the TemporaryFile instance
    will not guaranteed to be visible in the filesystem and is immediately removed
    once closed. This class will only delete files or directories upon request or
    upon garbage collection.

    Ensure this class does not go out of scope without other references to it
    unless you don't want to use the files any more.
    """

    def __init__(self):
        """Initialize the AutoDeleteTempfileGenerator."""
        self.tempdirs = []
        self.tempfiles = []

    def __del__(self):
        """Delete all temporary files and directories."""
        self.delete_all()

    def mkstemp(self, suffix='', prefix='tmp', directory=None):
        """Create a temporary file managed by this class instance."""
        handle, fname = tempfile.mkstemp(suffix, prefix, directory)
        self.tempfiles.append(fname)
        return (handle, fname)

    def mkftemp(self, suffix='', prefix='tmp', directory=None, mode='w'):
        """Wrap self.mkstemp() to open the OS-level file handle.

        Open it as a normal Python handle with the given mode.
        """
        handle, fname = self.mkstemp(suffix, prefix, directory)
        handle = os.fdopen(handle, mode)
        return (handle, fname)

    def mkdtemp(self, suffix='', prefix='tmp', directory=None):
        """Create a temporary directory managed by this class instance."""
        fname = tempfile.mkdtemp(suffix, prefix, directory)
        self.tempdirs.append(fname)
        return fname

    def delete_all(self):
        """Force-delete all files and directories created by this instance.

        The class instance may be used without restriction after this call.
        """
        #
        for filename in self.tempfiles:
            if os.path.isfile(filename):
                os.remove(filename)
        # Remove directories via shutil
        for tempdir in self.tempdirs:
            if os.path.isdir(tempdir) or os.path.isfile(tempdir):
                shutil.rmtree(tempdir)
        # Remove files from list
        self.tempfiles = []
        self.tempdirs = []

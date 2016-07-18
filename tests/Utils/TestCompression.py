#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from subprocess import check_output
from nose.tools import assert_equal, assert_true, raises
from UliEngineering.Utils.Compression import *
from UliEngineering.Utils.Temporary import *

class TestAutoOpen(object):
    def __init__(self):
        # Use auto-managed temporary files
        self.tempfiles = AutoDeleteTempfileGenerator()
        self.tempdir = self.tempfiles.mkdtemp()

    def testAutoOpen(self):
        gzfile = self.tempdir + "/test.gz"
        bzfile = self.tempdir + "/test.bz2"
        xzfile = self.tempdir + "/test.xz"
        # Create files
        check_output("echo abc | gzip -c > {0}".format(gzfile), shell=True)
        check_output("echo def | bzip2 -c > {0}".format(bzfile), shell=True)
        check_output("echo ghi | xz -c > {0}".format(xzfile), shell=True)
        # Check output
        with auto_open(gzfile) as infile:
            assert_equal("abc\n", infile.read())
        with auto_open(bzfile) as infile:
            assert_equal("def\n", infile.read())
        with auto_open(xzfile) as infile:
            assert_equal("ghi\n", infile.read())




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from subprocess import check_output
from UliEngineering.Utils.Compression import *
from UliEngineering.Utils.Temporary import *
import unittest

class TestAutoOpen(unittest.TestCase):
    def setUp(self):
        # Use auto-managed temporary files
        self.tempfiles = AutoDeleteTempfileGenerator()
        self.tempdir = self.tempfiles.mkdtemp()

    def testAutoOpen(self):
        gzfile = os.path.join(self.tempdir, "test.gz")
        bzfile = os.path.join(self.tempdir, "test.bz2")
        xzfile = os.path.join(self.tempdir, "test.xz")
        # Create files
        check_output("echo abc | gzip -c > {0}".format(gzfile), shell=True)
        check_output("echo def | bzip2 -c > {0}".format(bzfile), shell=True)
        check_output("echo ghi | xz -c > {0}".format(xzfile), shell=True)
        # Check output
        with auto_open(gzfile) as infile:
            self.assertEqual("abc\n", infile.read())
        with auto_open(bzfile) as infile:
            self.assertEqual("def\n", infile.read())
        with auto_open(xzfile) as infile:
            self.assertEqual("ghi\n", infile.read())

    def testInvalidExtension(self):
        "Test auto_open with a .foo file"
        # No need to actually write the file!
        with self.assertRaises(ValueError):
            filename = os.path.join(self.tempdir, "test.foo")
            auto_open(filename)

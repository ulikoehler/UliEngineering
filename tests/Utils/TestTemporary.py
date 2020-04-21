#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
from UliEngineering.Utils.Temporary import *
import unittest

class TestTemporary(unittest.TestCase):
    def testMkstemp(self):
        tgen = AutoDeleteTempfileGenerator()
        # Create file and check if it exists
        (_, fname) = tgen.mkstemp()
        self.assertTrue(os.path.isfile(fname))
        # Delete and check if file has vanished
        tgen.delete_all()
        self.assertFalse(os.path.isfile(fname))


    def testMkftemp(self):
        tgen = AutoDeleteTempfileGenerator()
        # Create file and check if it exists
        (handle, fname) = tgen.mkftemp()
        # Test if we can do stuff with the file as with any open()ed file
        handle.write("foo")
        handle.close()
        # Should not be deleted on close
        self.assertTrue(os.path.isfile(fname))
        # Delete and check if file has vanished
        tgen.delete_all()
        self.assertFalse(os.path.isfile(fname))

    def testMkdtemp(self):
        tgen = AutoDeleteTempfileGenerator()
        # Create file and check if it exists
        dirname = tgen.mkdtemp()
        self.assertTrue(os.path.isdir(dirname))
        # Delete and check if file has vanished
        tgen.delete_all()
        self.assertFalse(os.path.isdir(dirname))


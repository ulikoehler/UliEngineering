#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
from nose.tools import assert_equal, assert_true, assert_false
from UliEngineering.Utils.Temporary import *

class TestTemporary(object):
    def testMkstemp(self):
        tgen = AutoDeleteTempfileGenerator()
        # Create file and check if it exists
        (_, fname) = tgen.mkstemp()
        assert_true(os.path.isfile(fname))
        # Delete and check if file has vanished
        tgen.delete_all()
        assert_false(os.path.isfile(fname))


    def testMkftemp(self):
        tgen = AutoDeleteTempfileGenerator()
        # Create file and check if it exists
        (handle, fname) = tgen.mkftemp()
        # Test if we can do stuff with the file as with any open()ed file
        handle.write("foo")
        handle.close()
        # Should not be deleted on close
        assert_true(os.path.isfile(fname))
        # Delete and check if file has vanished
        tgen.delete_all()
        assert_false(os.path.isfile(fname))

    def testMkdtemp(self):
        tgen = AutoDeleteTempfileGenerator()
        # Create file and check if it exists
        dirname = tgen.mkdtemp()
        assert_true(os.path.isdir(dirname))
        # Delete and check if file has vanished
        tgen.delete_all()
        assert_false(os.path.isdir(dirname))


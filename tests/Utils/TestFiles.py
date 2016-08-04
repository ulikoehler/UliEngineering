#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false
from UliEngineering.Utils.Files import *
from UliEngineering.Utils.Temporary import *

class TestFiles(object):

    def testCountLinesFilelike(self):
        assert_equal(1, count_lines(io.StringIO("foo")))
        assert_equal(1, count_lines(io.StringIO("foo\n")))
        assert_equal(1, count_lines(io.StringIO("foo\n\n")))
        assert_equal(1, count_lines(io.StringIO("foo\n\n\n")))
        assert_equal(2, count_lines(io.StringIO("foo\n\n f\n")))
        assert_equal(2, count_lines(io.StringIO("foo\na\n\n")))
        assert_equal(2, count_lines(io.StringIO("foo\n\n\na")))

        assert_equal(3, count_lines(io.StringIO("foo\r\n\r\n\r\n\na\r\na\n\n\n\r\n\r\r\n")))

    def testCountLinesTempfile(self):
        tmp = AutoDeleteTempfileGenerator()
        # Test 1
        handle, fname = tmp.mkftemp()
        handle.write("foo\n\n\na")
        handle.close()
        assert_equal(2, count_lines(fname))
        # Test 2
        handle, fname = tmp.mkftemp()
        handle.write("foo\r\n\r\n\r\n\na\r\na\n\n\n\r\n\r\r\n")
        handle.close()
        assert_equal(3, count_lines(fname))

class TestColumnExtraction(object):
    def testExtractColumn(self):
        tmp = AutoDeleteTempfileGenerator()
        handle, fname = tmp.mkftemp()
        handle.write("foo\nbar\n\na")
        handle.close()
        # Read back
        assert_equal(extract_column(fname), ["foo", "bar", "a"])

    def testExtractNumericColumn(self):
        tmp = AutoDeleteTempfileGenerator()
        handle, fname = tmp.mkftemp()
        handle.write("3.2\n2.4\n\n1.5")
        handle.close()
        # Read back
        assert_allclose(extract_numeric_column(fname), [3.2, 2.4, 1.5])


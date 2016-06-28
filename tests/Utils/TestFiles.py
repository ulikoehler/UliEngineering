#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from nose.tools import assert_equal, assert_true, assert_false
from UliEngineering.Utils.Files import *

class TestFiles(object):
    def testCountLines(self):
        assert_equal(1, count_lines(io.StringIO("foo")))
        assert_equal(1, count_lines(io.StringIO("foo\n")))
        assert_equal(1, count_lines(io.StringIO("foo\n\n")))
        assert_equal(1, count_lines(io.StringIO("foo\n\n\n")))
        assert_equal(2, count_lines(io.StringIO("foo\n\n f\n")))
        assert_equal(2, count_lines(io.StringIO("foo\na\n\n")))
        assert_equal(2, count_lines(io.StringIO("foo\n\n\na")))

        assert_equal(3, count_lines(io.StringIO("foo\r\n\r\n\r\n\na\r\na\n\n\n\r\n\r\r\n")))

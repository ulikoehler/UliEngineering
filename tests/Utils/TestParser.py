#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true, raises
from UliEngineering.Utils.Parser import *


class TestParseIntFloat(object):
    def test_parse_int_or_float(self):
        assert_equal(parse_int_or_float("1"), 1)
        assert_equal(parse_int_or_float("1.0"), 1.0)
        assert_equal(parse_int_or_float("-2.225"), -2.225)

    @raises(ValueError)
    def test_parse_int_or_float_err(self):
        parse_int_or_float("3ahtj4")

    def test_try_parse_int_or_float(self):
        assert_equal(try_parse_int_or_float("1"), 1)
        assert_equal(try_parse_int_or_float("1.0"), 1.0)
        assert_equal(try_parse_int_or_float("-2.225"), -2.225)
        assert_equal(try_parse_int_or_float("bx3613"), "bx3613")
    
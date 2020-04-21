#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import self.assertEqual, assert_true, raises
from UliEngineering.Utils.Parser import *
import unittest

class TestParseIntFloat(unittest.TestCase):
    def test_parse_int_or_float(self):
        self.assertEqual(parse_int_or_float("1"), 1)
        self.assertEqual(parse_int_or_float("1.0"), 1.0)
        self.assertEqual(parse_int_or_float("-2.225"), -2.225)

    def test_parse_int_or_float_err(self):
        with self.assertRaises(ValueError):
            parse_int_or_float("3ahtj4")

    def test_try_parse_int_or_float(self):
        self.assertEqual(try_parse_int_or_float("1"), 1)
        self.assertEqual(try_parse_int_or_float("1.0"), 1.0)
        self.assertEqual(try_parse_int_or_float("-2.225"), -2.225)
        self.assertEqual(try_parse_int_or_float("bx3613"), "bx3613")
    
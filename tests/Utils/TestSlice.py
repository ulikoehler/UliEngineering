#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true, raises
from UliEngineering.Utils.Slice import *
import unittest

class TestShiftSlice(unittest.TestCase):
    def test_shift_slice(self):
        assert_equal(shift_slice(slice(9123, 10000), by=0), slice(9123, 10000))
        assert_equal(shift_slice(slice(9123, 10000), by=10), slice(9133, 10010))
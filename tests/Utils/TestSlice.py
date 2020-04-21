#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Utils.Slice import *
import unittest

class TestShiftSlice(unittest.TestCase):
    def test_shift_slice(self):
        self.assertEqual(shift_slice(slice(9123, 10000), by=0), slice(9123, 10000))
        self.assertEqual(shift_slice(slice(9123, 10000), by=10), slice(9133, 10010))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.Tolerance import *
from UliEngineering.EngineerIO import auto_format
from UliEngineering.Utils.Range import ValueRange
import numpy as np
import unittest

class TestValueRangeOverTolerance(unittest.TestCase):
    def test_value_range_over_tolerance(self):
        # Test with simple ppm input
        self.assertEqual(str(value_range_over_tolerance("1 kΩ", "1 %")),
            str(ValueRange(990, 1010, "Ω"))
        )
        self.assertEqual(str(value_range_over_tolerance("1 kΩ", "1000 ppm")),
            str(ValueRange(999., 1001.0, "Ω"))
        )
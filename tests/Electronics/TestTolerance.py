#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Electronics.Tolerance import value_range_over_tolerance
from UliEngineering.Utils.Range import ValueRange
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
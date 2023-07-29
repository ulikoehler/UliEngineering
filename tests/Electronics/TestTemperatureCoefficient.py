#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.TemperatureCoefficient import *
from UliEngineering.EngineerIO import auto_format
from UliEngineering.Utils.Range import ValueRange
import numpy as np
import unittest

class TestTemperatureCoefficient(unittest.TestCase):
    def test_value_range_over_temperature_zero(self):
        # Test with simple ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "0 ppm")),
            str(ValueRange(1000, 1000, "Ω"))
        )

    def test_value_range_over_temperature1(self):
        # Test with simple ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "100 ppm")),
            str(ValueRange(994, 1006, "Ω"))
        )
    
    def test_value_range_over_temperature2(self):
        # Test with +- the same ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", ("-100 ppm", "100 ppm"))),
            str(ValueRange(994, 1006, "Ω"))
        )
        # Test with ++ the same ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", ("+100 ppm", "+100 ppm"))),
            str(ValueRange(994, 1006, "Ω"))
        )
    
    def test_value_range_over_temperature3(self):
        # Test with +- the same ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", ("0 ppm", "100 ppm"))),
            str(ValueRange(994, 1006, "Ω"))
        )

    def test_value_range_over_temperature_percent(self):
        # Test with +- the same ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "1 %")),
            str(ValueRange(350, 1650, "Ω"))
        )
        # Test with +- the same ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "1.006 %")),
            str(ValueRange(346, 1654, "Ω"))
        )
    
    def test_value_range_over_temperature_tolerance(self):
        # Test with +- the same ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "100 ppm", tolerance="1%")),
            str(ValueRange(984, 1017, "Ω"))
        )
        # Test with ++ the same ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", ("-100 ppm", "+100 ppm"), tolerance=("-0%", "+1%"))),
            str(ValueRange(994, 1017, "Ω"))
        )

class TestValueAtTemperature(unittest.TestCase):
    def test_value_at_temperature(self):
        # Ref temp => zero difference
        assert_approx_equal(value_at_temperature("1 kΩ", "25 °C", "100 ppm"), 1000.0)
        # delta T = 10° => 10 * 100 ppm
        assert_approx_equal(value_at_temperature("1 kΩ", "35 °C", "100 ppm"), 1001.)
        # delta T = -10° => -10 * 100 ppm
        assert_approx_equal(value_at_temperature("1 kΩ", "15 °C", "100 ppm"), 999.)

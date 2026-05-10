#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Electronics.Crystal import load_capacitors, actual_load_capacitance, crystal_deviation_seconds_per_minute, crystal_deviation_seconds_per_hour, crystal_deviation_seconds_per_day, crystal_deviation_seconds_per_month, crystal_deviation_seconds_per_year, normalize_ppm, PPM
from UliEngineering.EngineerIO import auto_format
import unittest

class TestCrystal(unittest.TestCase):
    def test_load_capacitor(self):
        # Example from https://blog.adafruit.com/2012/01/24/choosing-the-right-crystal-and-caps-for-your-design/
        self.assertEqual(auto_format(load_capacitors, "6 pF", cpin="3 pF", cstray="2pF"), '5.00 pF')

    def test_actual_load_capacitance(self):
        self.assertEqual(auto_format(actual_load_capacitance, "5 pF", cpin="3 pF", cstray="2pF"), '6.00 pF')

    def test_deviation(self):
        self.assertEqual(auto_format(crystal_deviation_seconds_per_minute, "20 ppm"), '1.20 ms')
        self.assertEqual(auto_format(crystal_deviation_seconds_per_hour, "20 ppm"), '72.0 ms')
        self.assertEqual(auto_format(crystal_deviation_seconds_per_day, "20 ppm"), '1.73 s')
        self.assertEqual(auto_format(crystal_deviation_seconds_per_month, "20 ppm"), '53.6 s')
        self.assertEqual(auto_format(crystal_deviation_seconds_per_year, "20 ppm"), '631 s')

class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(PPM)

    def test_normalize_ppm_various_units(self):
        """Test normalize_ppm with various unit inputs"""
        test_cases = [
            ("1 ppm", 1e-6),
            ("1 ppb", 1e-9),
            ("1 ppt", 1e-12),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_ppm(input_val)
                self.assertAlmostEqual(result, expected)

    def test_crystal_functions_various_units(self):
        """Test crystal functions with various unit inputs"""
        # Test deviation functions with different ppm units
        d1 = crystal_deviation_seconds_per_minute("20 ppm")
        d2 = crystal_deviation_seconds_per_minute("20000 ppb")
        self.assertAlmostEqual(d1, d2)

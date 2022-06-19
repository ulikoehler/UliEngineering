#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Electronics.Crystal import *
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

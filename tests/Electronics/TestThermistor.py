#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Electronics.Thermistor import *
import unittest

class TestThermistor(unittest.TestCase):
    def test_thermistor_b_value_numeric(self):
        # Test case 1: Numeric inputs
        r1 = 1000  # 1kΩ
        r2 = 10000  # 10kΩ
        t1 = 25.0  # 25°C
        t2 = 100.0  # 100°C
        t1_kelvin = t1 + 273.15
        t2_kelvin = t2 + 273.15
        expected_result = (t1_kelvin * t2_kelvin) / (t2_kelvin - t1_kelvin) * np.log(r1 / r2)
        self.assertAlmostEqual(thermistor_b_value(r1, r2, t1, t2), expected_result)

    def test_thermistor_b_value_kelvin(self):
        # Test case 2: Inputs with temperature in Kelvin
        r1 = 1000  # 1kΩ
        r2 = 10000  # 10kΩ
        t1 = "298.15K"  # 25°C in Kelvin
        t2 = "373.15K"  # 100°C in Kelvin
        t1_numeric = 25.0
        t2_numeric = 100.0
        t1_kelvin = t1_numeric + 273.15
        t2_kelvin = t2_numeric + 273.15
        expected_result = (t1_kelvin * t2_kelvin) / (t2_kelvin - t1_kelvin) * np.log(r1 / r2)
        self.assertAlmostEqual(thermistor_b_value(r1, r2, t1, t2), expected_result)

    def test_thermistor_b_value_str_resistance(self):
        # Test case 3: Inputs with resistance in string format
        r1 = "1kΩ"
        r2 = "10kΩ"
        t1 = 25.0  # 25°C
        t2 = 100.0  # 100°C
        t1_kelvin = t1 + 273.15
        t2_kelvin = t2 + 273.15
        expected_result = (t1_kelvin * t2_kelvin) / (t2_kelvin - t1_kelvin) * np.log(1000 / 10000)
        self.assertAlmostEqual(thermistor_b_value(r1, r2, t1, t2), expected_result)

    def test_thermistor_b_value_fahrenheit(self):
        # Test case 4: Inputs with temperature in Fahrenheit
        r1 = 1000  # 1kΩ
        r2 = 10000  # 10kΩ
        t1 = "77°F"  # 25°C
        t2 = "212°F"  # 100°C
        t1_numeric = 25.0
        t2_numeric = 100.0
        t1_kelvin = t1_numeric + 273.15
        t2_kelvin = t2_numeric + 273.15
        expected_result = (t1_kelvin * t2_kelvin) / (t2_kelvin - t1_kelvin) * np.log(r1 / r2)
        self.assertAlmostEqual(thermistor_b_value(r1, r2, t1, t2), expected_result)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import self.assertEqual, assert_true
from UliEngineering.Electronics.Resistors import *
from UliEngineering.EngineerIO import *
import unittest

class TestResistors(unittest.TestCase):
    def test_parallel_resistors(self):
        assert_approx_equal(parallel_resistors(1000.0, 1000.0), 500.0)
        assert_approx_equal(parallel_resistors(1000.0, 1000.0, 500.0), 250.0)
        assert_approx_equal(parallel_resistors("1kΩ", "1kΩ"), 500.0)

    def test_series_resistors(self):
        assert_approx_equal(series_resistors(1000.0, 1000.0), 2000.0)
        assert_approx_equal(series_resistors(1000.0, 1000.0, 500.0), 2500.0)
        assert_approx_equal(series_resistors("1kΩ", "1kΩ"), 2000.0)

    def test_standard_resistors(self):
        assert_true(len(list(standard_resistors())) > 500)

    def test_find_nearest_resistor(self):
        self.assertEqual(nearest_resistor(5000, sequence=e48), 5110.0)
        self.assertEqual(nearest_resistor(4998), 4990.0)

    def test_current_through_resistor(self):
        assert_approx_equal(current_through_resistor("1k", "1V"), 1e-3)
        assert_approx_equal(current_through_resistor(1e3, 2), 2e-3)
        assert_approx_equal(current_through_resistor("1Ω", "2V"), 2)

    def test_resistor_by_voltage_and_current(self):
        assert_approx_equal(resistor_by_voltage_and_current("2.5 V", "1 uA"), 2.5e6)
        self.assertEqual(auto_format(resistor_by_voltage_and_current, "2.5 V", "1 uA"), "2.50 MΩ")


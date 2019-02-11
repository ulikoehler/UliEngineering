#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal
from UliEngineering.Electronics.VoltageDivider import *
from UliEngineering.EngineerIO import auto_format

class TestNoiseDensity(object):
    def test_unloaded_ratio(self):
        assert_approx_equal(voltage_divider_ratio(1000.0, 1000.0), 0.5)
        # Top resistor has lower value => ratio < 0.5
        assert_approx_equal(voltage_divider_ratio(600, 1000.0), 0.625)

    def test_loaded_ratio(self):
        assert_approx_equal(voltage_divider_ratio(1000.0, 1000.0, 1e60), 0.5)
        assert_approx_equal(voltage_divider_ratio(1000.0, 1000.0, 1000.0), 0.6666666666666666)
        assert_approx_equal(voltage_divider_ratio("1kΩ", "1kΩ", "10 MΩ"), 0.500024998)

    def test_bottom_resistor_by_ratio(self):
        assert_approx_equal(bottom_resistor_by_ratio(1000.0, 0.5), 1000.0)
        assert_approx_equal(bottom_resistor_by_ratio(200.0, 5/6.0), 1000.0)
        assert_equal(auto_format(bottom_resistor_by_ratio, 1000.0, 0.5), "1000 Ω")
        assert_equal(auto_format(bottom_resistor_by_ratio, 400.0, 5/6.0), "2.00 kΩ")

    def test_top_resistor_by_ratio(self):
        assert_approx_equal(top_resistor_by_ratio(1000.0, 0.5), 1000.0)
        assert_approx_equal(top_resistor_by_ratio(1000.0, 5/6.0), 200.0)
        assert_equal(auto_format(top_resistor_by_ratio, 1000.0, 0.5), "1000 Ω")
        assert_equal(auto_format(top_resistor_by_ratio, 1000.0, 5/6.0), "200 Ω")

    def test_feedback_resistors(self):
        assert_approx_equal(feedback_top_resistor(1.8, 816e3, 0.8), 1020e3)
        assert_approx_equal(feedback_bottom_resistor(1.8, 1020e3, 0.8), 816e3)
        # Test string input
        assert_approx_equal(feedback_top_resistor("1.8 V", "816 kΩ", "0.8 V"), 1020e3)
        assert_approx_equal(feedback_bottom_resistor("1.8 V", "1020 kΩ", "0.8 V"), 816e3)

    def test_feedback_voltage(self):
        assert_approx_equal(feedback_actual_voltage(1020e3, 816e3, 0.8), 1.8)
        # String input
        assert_approx_equal(feedback_actual_voltage("1.02 MΩ", "816 kΩ", "0.8 V"), 1.8)

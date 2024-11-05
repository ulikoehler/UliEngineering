#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Electronics.LED import *
from UliEngineering.Exceptions import OperationImpossibleException
from UliEngineering.EngineerIO import auto_format
import unittest
import pytest

class TestLEDSeriesResistors(unittest.TestCase):
    def test_led_series_resistor(self):
        # Example verified at http://www.elektronik-kompendium.de/sites/bau/1109111.htm
        # Also verified at https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-led-series-resistor
        assert_approx_equal(led_series_resistor(12.0, 20e-3, 1.6), 520.)
        assert_approx_equal(led_series_resistor("12V", "20 mA", "1.6V"), 520.)
        assert_approx_equal(led_series_resistor(12.0, 20e-3, LEDForwardVoltages.Red), 520.)

    def test_led_series_resistor_invalid(self):
        # Forward voltage too high for supply voltage
        with self.assertRaises(OperationImpossibleException):
            assert_approx_equal(led_series_resistor("1V", "20 mA", "1.6V"), 520.)

    def test_led_series_resistor_power(self):
        # Values checked using https://www.pollin.de/led-vorwiderstands-rechner
        self.assertEqual(auto_format(led_series_resistor_power, "5V", "20mA", "2V"), "60.0 mW")
        self.assertEqual(auto_format(led_series_resistor_power, "5V", "20mA", "3V"), "40.0 mW")
        self.assertEqual(auto_format(led_series_resistor_power, "5V", "10mA", "2V"), "30.0 mW")
        self.assertEqual(auto_format(led_series_resistor_power, "5V", "10mA", "3V"), "20.0 mW")
        self.assertEqual(auto_format(led_series_resistor_power, "12V", "10mA", "2V"), "100 mW")
        
    def test_led_series_resistor_power_invalid(self):
        with pytest.raises(OperationImpossibleException):
            led_series_resistor_power("2V", "20mA", "3V")

    def test_led_series_resistor_maximum_current(self):
        # Test with valid inputs
        # Verified using https://www.omnicalculator.com/physics/ohms-law
        assert_approx_equal(led_series_resistor_maximum_current(10, 0.25), 0.1581139)
        assert_approx_equal(led_series_resistor_maximum_current(1, 2.56), 1.6)

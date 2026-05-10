#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Electronics.LED import led_series_resistor, LEDForwardVoltages, led_series_resistor_power, led_series_resistor_maximum_current
from UliEngineering.Electronics.Diode import VoltageV, CurrentA, ResistanceOhm, PowerW
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

class TestAnnotatedTypes(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the type annotations are available"""
        self.assertIsNotNone(VoltageV)
        self.assertIsNotNone(CurrentA)
        self.assertIsNotNone(ResistanceOhm)
        self.assertIsNotNone(PowerW)

    def test_led_functions_various_units(self):
        """Test LED functions with various unit inputs"""
        # Test led_series_resistor with different unit representations
        r1 = led_series_resistor("12 V", "20 mA", "1.6 V")
        r2 = led_series_resistor("12000 mV", "0.02 A", "1600 mV")
        self.assertAlmostEqual(r1, r2)

        # Test led_series_resistor_power with different units
        p1 = led_series_resistor_power("5 V", "20 mA", "2 V")
        p2 = led_series_resistor_power("5000 mV", "0.02 A", "2000 mV")
        self.assertAlmostEqual(p1, p2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.Temperature import *
from UliEngineering.Exceptions import *
from UliEngineering.EngineerIO import auto_format
import unittest

class TestTemperature(unittest.TestCase):
    def testNormalizeTemperature(self):
        # Pure numbers, assumed to be °C
        with self.assertRaises(InvalidUnitException):
            assert_approx_equal(normalize_temperature("0"), 273.15)
        with self.assertRaises(InvalidUnitException):
            assert_approx_equal(normalize_temperature("1"), 274.15)
        assert_approx_equal(normalize_temperature(1), 274.15)
        assert_approx_equal(normalize_temperature(1, default_unit="K"), 1.0)
        # With units
        assert_approx_equal(normalize_temperature("1 C"), 274.15)
        assert_approx_equal(normalize_temperature("1 °C"), 274.15)
        assert_approx_equal(normalize_temperature("1°C"), 274.15)
        assert_approx_equal(normalize_temperature("1 K"), 1.0)
        assert_approx_equal(normalize_temperature("60 F"), 288.71, significant=5)
        # Signs
        assert_approx_equal(normalize_temperature("-1°C"), 272.15)
        assert_approx_equal(normalize_temperature("-200°C"), 73.15)

    def testNormalizeTemperatureCelsius(self):
        assert_approx_equal(normalize_temperature_celsius("-200°C"), -200.0)
        assert_approx_equal(normalize_temperature_celsius("273.15 K"), 0.0)

    def testAutoFormatTemperature(self):
        self.assertEqual(auto_format(normalize_temperature, "-200°C"), "73.1 K")
        self.assertEqual(auto_format(normalize_temperature_celsius, "-200°C"), "-200.00 °C")
        self.assertEqual(auto_format(normalize_temperature_celsius, "-111 °C"), "-111.00 °C")
        self.assertEqual(auto_format(normalize_temperature_celsius, "0 °K"), "-273.15 °C")
        self.assertEqual(auto_format(normalize_temperature_celsius, "0 K"), "-273.15 °C")

    def test_temperature_with_dissipation(self):
        assert_approx_equal(temperature_with_dissipation("1 W", "1 °C/W"), 26.)
        assert_approx_equal(temperature_with_dissipation("1 W", "50 °C/W"), 25. + 50.)

    def testWrongUnit(self):
        with self.assertRaises(InvalidUnitException):
            normalize_temperature("150V")

    def testInvalidUnit(self):
        with self.assertRaises(ValueError):
            normalize_temperature("1G50 G")

class TestTemperatureConversion(unittest.TestCase):
    def test_fahrenheit_to_celsius(self):
        assert_approx_equal(fahrenheit_to_celsius("11 °F"), -11.66666667)
        assert_approx_equal(fahrenheit_to_celsius(11.), -11.666666667)
        assert_approx_equal(fahrenheit_to_celsius("20 °F"), -6.66666667)
        assert_approx_equal(fahrenheit_to_celsius(20.), -6.666666667)
        self.assertEqual(auto_format(fahrenheit_to_celsius, "20 °F"), "-6.67 °C")

    def test_fahrenheit_to_kelvin(self):
        assert_approx_equal(fahrenheit_to_kelvin("11 °F"), 261.483333333)
        assert_approx_equal(fahrenheit_to_kelvin(11.), 261.483333333)
        assert_approx_equal(fahrenheit_to_kelvin("20 °F"), 266.483333333)
        assert_approx_equal(fahrenheit_to_kelvin(20.), 266.483333333)
        self.assertEqual(auto_format(fahrenheit_to_kelvin, "20 °F"), "266 K")

    def test_kelvin_to_celsius(self):
        assert_approx_equal(kelvin_to_celsius("11 K"), -262.15)
        assert_approx_equal(kelvin_to_celsius(11.), -262.15)
        assert_approx_equal(kelvin_to_celsius("300 K"), 26.85)
        assert_approx_equal(kelvin_to_celsius(300.), 26.85)
        self.assertEqual(auto_format(kelvin_to_celsius, "300 K"), "26.9 °C")

    def test_celsius_to_kelvin(self):
        assert_approx_equal(celsius_to_kelvin("-262.15 °C"), 11.)
        assert_approx_equal(celsius_to_kelvin(-262.15), 11.)
        assert_approx_equal(celsius_to_kelvin("26.85 °C"), 300.)
        assert_approx_equal(celsius_to_kelvin(26.85), 300.)
        self.assertEqual(auto_format(celsius_to_kelvin, "26.85 °C"), "300 K")



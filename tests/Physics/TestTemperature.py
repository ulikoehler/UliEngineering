#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import raises, assert_equal
from UliEngineering.Physics.Temperature import *
from UliEngineering.Exceptions import *
from UliEngineering.EngineerIO import auto_format

class TestTemperature(object):
    def testNormalizeTemperature(self):
        # Pure numbers
        assert_approx_equal(normalize_temperature("0"), 273.15)
        assert_approx_equal(normalize_temperature("1"), 274.15)
        assert_approx_equal(normalize_temperature(1), 274.15)
        assert_approx_equal(normalize_temperature(1, default_unit="°K"), 1.0)
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
        assert_approx_equal(normalize_temperature_celsius("273.15 °K"), 0.0)

    def testAutoFormatTemperature(self):
        assert_equal(auto_format(normalize_temperature, "-200°C"), "73.1 °K")
        assert_equal(auto_format(normalize_temperature_celsius, "-200°C"), "-200.00 °C")
        assert_equal(auto_format(normalize_temperature_celsius, "-111 °C"), "-111.00 °C")
        assert_equal(auto_format(normalize_temperature_celsius, "0 °K"), "-273.15 °C")

    @raises(InvalidUnitException)
    def testWrongUnit(self):
        normalize_temperature("150V")

    @raises(ConversionException)
    def testInvalidUnit(self):
        normalize_temperature("150°G")

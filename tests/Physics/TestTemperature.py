#!/usr/bin/env python3
from numpy.testing import assert_approx_equal
from nose.tools import raises
from UliEngineering.Physics.Temperature import *
from UliEngineering.Exceptions import *

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
    @raises(InvalidUnitException)
    def testWrongUnit(self):
        normalize_temperature("150V")
    @raises(ConversionException)
    def testInvalidUnit(self):
        normalize_temperature("150°G")
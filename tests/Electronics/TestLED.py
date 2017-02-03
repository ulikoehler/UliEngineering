#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import raises
from UliEngineering.Electronics.LED import *
from UliEngineering.Exceptions import OperationImpossibleException
from UliEngineering.EngineerIO import auto_format

class TestLEDSeriesResistors(object):
    def test_led_series_resistor(self):
        # Example verified at http://www.elektronik-kompendium.de/sites/bau/1109111.htm
        # Also verified at https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-led-series-resistor
        assert_approx_equal(led_series_resistor(12.0, 20e-3, 1.6), 520.)
        assert_approx_equal(led_series_resistor("12V", "20 mA", "1.6V"), 520.)
        assert_approx_equal(led_series_resistor(12.0, 20e-3, LEDForwardVoltages.Red), 520.)

    @raises(OperationImpossibleException)
    def test_led_series_resistor_invalid(self):
        # Forward voltage too high for supply voltage
        assert_approx_equal(led_series_resistor("1V", "20 mA", "1.6V"), 520.)

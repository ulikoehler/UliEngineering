#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Electronics.LED import *
from UliEngineering.EngineerIO import auto_format

class TestLEDSeriesResistors(object):
    def test_led_series_resistor(self):
        # Example verified at http://www.elektronik-kompendium.de/sites/bau/1109111.htm
        # Also verified at https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-led-series-resistor
        assert_approx_equal(led_series_resistor(12.0, 20e-3, 1.6), 520.)
        assert_approx_equal(led_series_resistor("12V", "20 mA", "1.6V"), 520.)

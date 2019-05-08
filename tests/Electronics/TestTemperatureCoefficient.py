#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Electronics.TemperatureCoefficient import *
from UliEngineering.EngineerIO import auto_format
import numpy as np

class TestTemperatureCoefficient(object):
    def test_value_range_over_temperature1(self):
        # Test with simple ppm input
        assert_equal(value_range_over_temperature("1 kΩ", "100 ppm"),
            ValueRange("994 Ω", "1.006 kΩ")
        )
    
    def test_value_range_over_temperature2(self):
        # Test with +- the same ppm input
        assert_equal(value_range_over_temperature("1 kΩ", ("100 ppm", "100 ppm")),
            ValueRange("994 Ω", "1.006 kΩ")
        )
    
    def test_value_range_over_temperature3(self):
        # Test with +- the same ppm input
        assert_equal(value_range_over_temperature("1 kΩ", ("0 ppm", "100 ppm")),
            ValueRange("1000 Ω", "1.006 kΩ")
        )

    def test_value_range_over_temperature_percent(self):
        # Test with +- the same ppm input
        assert_equal(value_range_over_temperature("1 kΩ", "1 %"),
            ValueRange("350 Ω", "1.650 kΩ")
        )
        # Test with +- the same ppm input
        assert_equal(value_range_over_temperature("1 kΩ", "1.006 %"),
            ValueRange("346 Ω", "1.654 kΩ")
        )


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
            ValueRange("994 Ω", "1.01 kΩ")
        )

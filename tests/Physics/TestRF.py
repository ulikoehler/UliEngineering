#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Physics.RF import *
from UliEngineering.EngineerIO import auto_format
import numpy as np

class TestRF(object):
    def test_quality_factor(self):
        assert_approx_equal(quality_factor("8.000 MHz", "1 kHz"), 8000.0)
        assert_approx_equal(quality_factor("8.000 MHz", "1 MHz"), 8.0)

    def test_resonant_impedance(self):
        assert_approx_equal(resonant_impedance("100 uH", "10 nF", Q=30.0), 10./3)
        assert_equal(format_value(resonant_impedance("100 uH", "10 nF", Q=30.0), "Ω"), '3.33 Ω')

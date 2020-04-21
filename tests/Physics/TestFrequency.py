#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_less
from UliEngineering.Physics.Frequency import *
from UliEngineering.Exceptions import *
import functools
import numpy as np
import unittest

class TestFrequencies(unittest.TestCase):
    def test_frequency_to_period(self):
        assert_approx_equal(frequency_to_period(0.1), 10)
        assert_approx_equal(frequency_to_period("0.1 Hz"), 10)
        assert_approx_equal(frequency_to_period("10 Hz"), 0.1)
        assert_approx_equal(frequency_to_period("10 kHz"), 0.1e-3)

    def test_period_to_frequency(self):
        assert_approx_equal(frequency_to_period(10), 0.1)
        assert_approx_equal(frequency_to_period("10 s"), 0.1)
        assert_approx_equal(frequency_to_period("10 ks"), 0.1e-3)
        assert_approx_equal(frequency_to_period("1 ms"), 1e3)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.Physics.Frequency import frequency_to_period, normalize_frequency, normalize_rpm
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

    def test_normalize_frequency(self):
        assert_approx_equal(normalize_frequency("60 rpm"), 1.0)
        assert_approx_equal(normalize_frequency("1 kHz"), 1000.0)

    def test_normalize_frequency_iterables(self):
        assert_allclose(normalize_frequency(["60 rpm", "120 rpm"]), [1.0, 2.0])
        assert_allclose(normalize_frequency(item for item in ["60 rpm", "120 rpm"]), [1.0, 2.0])
        assert_allclose(normalize_rpm(["60 rpm", "120 rpm"]), [60.0, 120.0])

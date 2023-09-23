#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.Filter import *
import unittest

class TestFilter(unittest.TestCase):
    def test_lc_cutoff_frequency(self):
        # Test with string input
        assert_approx_equal(lc_cutoff_frequency("3.3uH", "22uF"), 18678.92254731818)
        # Test with numeric input
        assert_approx_equal(lc_cutoff_frequency(3.3e-6, 22e-6), 18678.92254731818)

    def test_rc_cutoff_frequency(self):
        """
        Test the rc_cutoff_frequency function
        """
        # Verified using https://www.omnicalculator.com/physics/low-pass-filter
        # Test with string input
        assert_approx_equal(rc_cutoff_frequency("124k", "100pF"), 12835.07605579801)
        # Test with numeric input
        assert_approx_equal(rc_cutoff_frequency(124e3, 100e-12), 12835.07605579801)

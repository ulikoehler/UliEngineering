#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Electronics.OpAmp import *
from UliEngineering.EngineerIO import auto_format
import unittest
import numpy as np

class TestOpAmp(unittest.TestCase):
    def test_summing_amplifier_noninv(self):
        v = summing_amplifier_noninv("2.5V", "500mV", "1kΩ", "1kΩ", "1kΩ", "1kΩ")
        assert_approx_equal(v, 3.0)
        v = summing_amplifier_noninv(2.5, 0.5, 1e3, 1e3, 300, 1e3)
        assert_approx_equal(v, 6.5)
        self.assertEqual(auto_format(summing_amplifier_noninv, 2.5, 0.5, 1e3, 1e3, 300, 1e3), "6.50 V")

    def test_noninverting_amplifier_gain(self):
        # Test cases checked using https://circuitdigest.com/calculators/op-amp-gain-calculator
        assert_approx_equal(noninverting_amplifier_gain("1kΩ", "1kΩ"), 2.0)
        assert_approx_equal(noninverting_amplifier_gain(1e3, 1e3), 2.0)
        assert_approx_equal(noninverting_amplifier_gain("2kΩ", "1kΩ"), 3.0)
        assert_approx_equal(noninverting_amplifier_gain(2e3, 1e3), 3.0)
        # Check auto_format
        self.assertEqual(auto_format(noninverting_amplifier_gain, 2e3, 1e3), "3.00 V/V")
        self.assertEqual(auto_format(noninverting_amplifier_gain, 1e3, 1e3), "2.00 V/V")
        self.assertEqual(auto_format(noninverting_amplifier_gain, "1kΩ", "1kΩ"), "2.00 V/V")
        # Test case with not-as-round numbers
        assert_approx_equal(noninverting_amplifier_gain("390kΩ", "15kΩ"), 27.0)
        assert_approx_equal(noninverting_amplifier_gain(390e3, 15e3), 27.0)
        # Test case with infinity resistor to GND (i.e. unity gain)
        assert_approx_equal(noninverting_amplifier_gain("1kΩ", np.inf), 1.0)
        assert_approx_equal(noninverting_amplifier_gain(1e3, np.inf), 1.0)
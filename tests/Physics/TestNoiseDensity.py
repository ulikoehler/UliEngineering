#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.NoiseDensity import *
from UliEngineering.EngineerIO import auto_format
import unittest

class TestNoiseDensity(unittest.TestCase):
    def testActualNoise(self):
        assert_approx_equal(actual_noise("100 µV", "100 Hz"), 1e-3)
        assert_approx_equal(actual_noise(1e-4, 100), 1e-3)
        self.assertEqual(auto_format(actual_noise, "100 µV", "100 Hz"), '1.00 mV')

    def testNoiseDensity(self):
        assert_approx_equal(noise_density("1.0 mV", "100 Hz"), 1e-4)
        assert_approx_equal(noise_density(1e-3, 100), 1e-4)
        self.assertEqual(auto_format(noise_density, "1.0 mV", "100 Hz"), '100 µV/√Hz')

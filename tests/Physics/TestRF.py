#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Physics.RF import *
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest

class TestRF(unittest.TestCase):
    def test_quality_factor(self):
        assert_approx_equal(quality_factor("8.000 MHz", "1 kHz"), 8000.0)
        assert_approx_equal(quality_factor("8.000 MHz", "1 MHz"), 8.0)

    def test_resonant_impedance(self):
        assert_approx_equal(resonant_impedance("100 uH", "10 nF", Q=30.0), 10./3)
        self.assertEqual(auto_format(resonant_impedance, "100 uH", "10 nF", Q=30.0), '3.33 Ω')

    def test_resonant_frequency(self):
        assert_approx_equal(resonant_frequency("100 uH", "10 nF"), 159154.94309189534)
        self.assertEqual(auto_format(resonant_frequency, "100 uH", "10 nF"), '159 kHz')

    def test_resonant_inductance(self):
        assert_approx_equal(resonant_inductance("250 kHz", "10 nF"), 4.052847345693511e-05)
        self.assertEqual(auto_format(resonant_inductance, "250 kHz", "10 nF"), '40.5 µH')

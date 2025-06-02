#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.Reactance import *
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest

class TestNoiseDensity(unittest.TestCase):
    def test_capacitive_reactance(self):
        assert_approx_equal(capacitive_reactance("100 pF", "3.2 MHz"), 497.3592)
        assert_approx_equal(capacitive_reactance(100e-12, 3.2e6), 497.3592)
        assert_approx_equal(capacitive_reactance(100e-12, 3.2e6), 497.3592)
        self.assertEqual(auto_format(capacitive_reactance, "100 pF", "3.2 MHz"), "497 Ω")

    def test_inductive_reactance(self):
        assert_approx_equal(inductive_reactance("100 µH", "3.2 MHz"), 2010.619)
        assert_approx_equal(inductive_reactance(100e-6, 3.2e6), 2010.619)
        self.assertEqual(auto_format(inductive_reactance, "100 µH", "3.2 MHz"), "2.01 kΩ")

    def test_numpy_arrays(self):
        l = np.asarray([100e-6, 200e-6])
        assert_allclose(inductive_reactance(l, 3.2e6), [2010.6193, 4021.23859659])

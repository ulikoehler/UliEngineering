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

    def test_inverse_inductive(self):
        # float inputs
        l = 100e-6
        f = 3.2e6
        x = inductive_reactance(l, f)
        l2 = inductance_from_reactance(x, f)
        assert_approx_equal(l2, l)

        # string inputs and formatting
        l_str = "100 µH"
        x = inductive_reactance(l_str, "3.2 MHz")
        self.assertAlmostEqual(x, 2010.619, places=3)
        l_back = inductance_from_reactance(x, "3.2 MHz")
        assert_approx_equal(l_back, 100e-6)
        af = auto_format(inductance_from_reactance, "2.01 kΩ", "3.2 MHz")
        # allow different float formatting like '100' or '100.0' etc.
        val, unit = af.split()
        self.assertAlmostEqual(float(val), 100.0, places=3)
        self.assertEqual(unit, "µH")

    def test_inverse_capacitive(self):
        # float input
        c = 100e-12
        f = 3.2e6
        x = capacitive_reactance(c, f)
        c2 = capacitance_from_reactance(x, f)
        assert_allclose(c2, c)

        # string inputs and formatting
        c_str = "100 pF"
        x = capacitive_reactance(c_str, "3.2 MHz")
        self.assertAlmostEqual(x, 497.3592, places=3)
        c_back = capacitance_from_reactance(x, "3.2 MHz")
        assert_approx_equal(c_back, 100e-12)
        af = auto_format(capacitance_from_reactance, "497 Ω", "3.2 MHz")
        val, unit = af.split()
        self.assertAlmostEqual(float(val), 100.0, places=3)
        self.assertEqual(unit, "pF")

    def test_inverse_numpy_arrays(self):
        # Inductive: arrays
        l = np.asarray([10e-6, 100e-6])
        f = 1e6
        x = inductive_reactance(l, f)
        l_back = inductance_from_reactance(x, f)
        assert_allclose(l_back, l)

        # Capacitive: arrays
        c = np.asarray([10e-12, 100e-12])
        x = capacitive_reactance(c, f)
        c_back = capacitance_from_reactance(x, f)
        assert_allclose(c_back, c)

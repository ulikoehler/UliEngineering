#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.Capacitors import *
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest

class TestCapacitors(unittest.TestCase):
    def test_capacitor_energy(self):
        assert_approx_equal(capacitor_energy("1.5 F", "5.0 V"), 18.75)
        assert_approx_equal(capacitor_energy("1.5 F", "0.0 V"), 0.0)
        self.assertEqual(auto_format(capacitor_energy, "100 mF", "1.2 V"), "72.0 mJ")

    def test_capacitor_charge(self):
        assert_approx_equal(capacitor_charge("1.5 F", "5.0 V"), 7.5)
        assert_approx_equal(capacitor_charge("1.5 F", "0.0 V"), 0.0)
        self.assertEqual(auto_format(capacitor_charge, "1.5 F", "5.0 V"), "7.50 C")

    def test_numpy_arrays(self):
        l = np.asarray([1.5, 0.1])
        assert_allclose(capacitor_energy(l, 5.0), [18.75, 1.25])

    def test_capacitor_lifetime(self):
        self.assertAlmostEqual(capacitor_lifetime(105, "2000 h", "105 °C", A=10), 2000)
        self.assertAlmostEqual(capacitor_lifetime(115, "2000 h", "105 °C", A=10), 1000)
        self.assertAlmostEqual(capacitor_lifetime(95, "2000 h", "105 °C", A=10), 4000)
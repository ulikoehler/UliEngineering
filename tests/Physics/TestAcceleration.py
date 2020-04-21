#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.Acceleration import *
from UliEngineering.EngineerIO import auto_format
import unittest
import scipy.constants

g0 = scipy.constants.physical_constants['standard acceleration of gravity'][0]

class TestAccelerationConversion(unittest.TestCase):
    def test_g_to_ms2(self):
        self.assertAlmostEqual(g_to_ms2(0.), 0)
        self.assertAlmostEqual(g_to_ms2(60.), 60 * g0)
        self.assertAlmostEqual(g_to_ms2(120.), 120 * g0)
        self.assertAlmostEqual(g_to_ms2(150.), 150 * g0)

    def test_ms2_to_g(self):
        self.assertAlmostEqual(ms2_to_g(0.), 0)
        self.assertAlmostEqual(ms2_to_g(60.), 60 / g0)
        self.assertAlmostEqual(ms2_to_g(120.), 120 / g0)
        self.assertAlmostEqual(ms2_to_g(150.), 150 / g0)

class TestCentrifugalAcceleration(unittest.TestCase):
    def test_centrifugal_acceleration(self):
        # Reference: https://techoverflow.net/2020/04/20/centrifuge-acceleration-calculator-from-rpm-and-diameter/
        # No acceleration if not turning
        self.assertAlmostEqual(centrifugal_acceleration(0.1, 0), 0.0, places=2)
        # These have speed > 0
        self.assertAlmostEqual(centrifugal_acceleration(0.1, 100), 39478.417, places=2)
        self.assertAlmostEqual(centrifugal_acceleration(0.2, 100), 78956.835, places=2)
        self.assertAlmostEqual(centrifugal_acceleration(0.2, 10), 789.568, places=2)

    def test_centrifuge_radius(self):
        # Reference: Inverse of test_centrifugal_acceleration()
        # Note: t
        # These have speed > 0
        self.assertAlmostEqual(centrifuge_radius(39478.417, 100), 0.1, places=2)
        self.assertAlmostEqual(centrifuge_radius(78956.835, 100), 0.2, places=2)
        self.assertAlmostEqual(centrifuge_radius(789.568, 10), 0.2, places=2)
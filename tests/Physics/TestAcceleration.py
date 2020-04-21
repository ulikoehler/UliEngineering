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


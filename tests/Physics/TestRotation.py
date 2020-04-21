#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.Rotation import *
from UliEngineering.EngineerIO import auto_format
import unittest

class TestRotationConversion(unittest.TestCase):
    def test_rpm_to_hz(self):
        self.assertAlmostEqual(rpm_to_Hz(0.), 0)
        self.assertAlmostEqual(rpm_to_Hz(60.), 1.)
        self.assertAlmostEqual(rpm_to_Hz(120.), 2.)
        self.assertAlmostEqual(rpm_to_Hz(150.), 2.5)

    def test_rpm_to_rps(self):
        self.assertAlmostEqual(rpm_to_rps(0.), 0)
        self.assertAlmostEqual(rpm_to_rps(60.), 1.)
        self.assertAlmostEqual(rpm_to_rps(120.), 2.)
        self.assertAlmostEqual(rpm_to_rps(150.), 2.5)

    def test_hz_to_rpm(self):
        self.assertAlmostEqual(hz_to_rpm(0.), 0)
        self.assertAlmostEqual(hz_to_rpm(1.), 60.)
        self.assertAlmostEqual(hz_to_rpm(2.), 120.)
        self.assertAlmostEqual(hz_to_rpm(2.5), 150.)

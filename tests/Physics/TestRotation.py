#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Physics.Rotation import *
import unittest
import math

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

class TestRotationOther(unittest.TestCase):
    def test_angular_speed(self):
        self.assertAlmostEqual(angular_speed(0.), 0)
        self.assertAlmostEqual(angular_speed("0 Hz"), 0)
        self.assertAlmostEqual(angular_speed(1), 1*2*math.pi)
        self.assertAlmostEqual(angular_speed("1 Hz"), 1*2*math.pi)
        self.assertAlmostEqual(angular_speed(1000), 1000*2*math.pi)
        self.assertAlmostEqual(angular_speed("1 kHz"), 1000*2*math.pi)

    def test_centrifugal_force(self):
        # Zero cases
        self.assertAlmostEqual(centrifugal_force(5, 10, 0), 0)
        self.assertAlmostEqual(centrifugal_force(0, 10, 500), 0)
        self.assertAlmostEqual(centrifugal_force(5, 0, 500), 0)
        # Non zero cases
        # Reference: https://www.thecalculator.co/others/Centrifugal-Force-Calculator-660.html
        self.assertAlmostEqual(centrifugal_force(5, 10, 500), 9869.604401089358, places=2)
        self.assertAlmostEqual(centrifugal_force("5", "10", "500"), 9869.604401089358, places=2)

    def test_rotation_linear_speed(self):
        self.assertAlmostEqual(rotation_linear_speed(1, 0.), 0)
        self.assertAlmostEqual(rotation_linear_speed(1, "0 Hz"), 0)
        self.assertAlmostEqual(rotation_linear_speed(1, 1), 1*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed(1, "1 Hz"), 1*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed(1, 1000), 1000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed(1, "1 kHz"), 1000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("1", "1 kHz"), 1000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("2", "1 kHz"), 2000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("1m", "1 kHz"), 1*2*math.pi)

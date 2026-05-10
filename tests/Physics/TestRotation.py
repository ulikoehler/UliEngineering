#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Units import InvalidUnitInContextException
from UliEngineering.Physics.Rotation import (
    angular_speed,
    centrifugal_force,
    hz_to_rpm,
    rotating_liquid_pressure,
    rotation_linear_speed,
    rpm_to_Hz,
    rpm_to_rps,
)
from UliEngineering.EngineerIO.Length import LengthMeters
from UliEngineering.Physics.Frequency import FrequencyHz, RotationRate
from UliEngineering.Physics.Mass import MassGrams
from UliEngineering.Physics.Density import DensityKgPerM3
import unittest
import math

class TestRotationConversion(unittest.TestCase):
    def test_rpm_to_hz(self):
        self.assertAlmostEqual(rpm_to_Hz("0 rpm"), 0)
        self.assertAlmostEqual(rpm_to_Hz("60 rpm"), 1.)
        self.assertAlmostEqual(rpm_to_Hz("120 rpm"), 2.)
        self.assertAlmostEqual(rpm_to_Hz("150 rpm"), 2.5)
        self.assertAlmostEqual(rpm_to_Hz("60 rpm"), 1.)

    def test_rpm_to_rps(self):
        self.assertAlmostEqual(rpm_to_rps("0 rpm"), 0)
        self.assertAlmostEqual(rpm_to_rps("60 rpm"), 1.)
        self.assertAlmostEqual(rpm_to_rps("120 rpm"), 2.)
        self.assertAlmostEqual(rpm_to_rps("150 rpm"), 2.5)

    def test_hz_to_rpm(self):
        self.assertAlmostEqual(hz_to_rpm(0.), 0)
        self.assertAlmostEqual(hz_to_rpm(1.), 60.)
        self.assertAlmostEqual(hz_to_rpm(2.), 120.)
        self.assertAlmostEqual(hz_to_rpm(2.5), 150.)
        self.assertAlmostEqual(hz_to_rpm("1 kHz"), 60000.)

class TestRotationOther(unittest.TestCase):
    def test_angular_speed(self):
        self.assertAlmostEqual(angular_speed(0.), 0)
        self.assertAlmostEqual(angular_speed("0 Hz"), 0)
        self.assertAlmostEqual(angular_speed(1), 1*2*math.pi)
        self.assertAlmostEqual(angular_speed("1 Hz"), 1*2*math.pi)
        self.assertAlmostEqual(angular_speed(1000), 1000*2*math.pi)
        self.assertAlmostEqual(angular_speed("1 kHz"), 1000*2*math.pi)
        self.assertAlmostEqual(angular_speed("60 rpm"), 1*2*math.pi)
        with self.assertRaises(InvalidUnitInContextException):
            angular_speed("1 m")

    def test_centrifugal_force(self):
        # Zero cases
        self.assertAlmostEqual(centrifugal_force(5, 10, 0), 0)
        self.assertAlmostEqual(centrifugal_force(0, 10, 500), 0)
        self.assertAlmostEqual(centrifugal_force(5, 0, 500), 0)
        # Non zero cases
        # Reference: https://www.thecalculator.co/others/Centrifugal-Force-Calculator-660.html
        self.assertAlmostEqual(centrifugal_force(5, 10, 500), 9869.604401089358, places=2)
        self.assertAlmostEqual(centrifugal_force("5", "10", "500"), 9869.604401089358, places=2)
        self.assertAlmostEqual(centrifugal_force("1 m", "60 rpm", "500 g"), 0.5 * (2 * math.pi) ** 2, places=6)
        self.assertAlmostEqual(centrifugal_force("1 m", "60 rpm", "0.5 kg"), 0.5 * (2 * math.pi) ** 2, places=6)

    def test_rotation_linear_speed(self):
        self.assertAlmostEqual(rotation_linear_speed(1, 0.), 0)
        self.assertAlmostEqual(rotation_linear_speed(1, "0 Hz"), 0)
        self.assertAlmostEqual(rotation_linear_speed(1, 1), 1*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed(1, "1 Hz"), 1*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed(1, 1000), 1000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed(1, "1 kHz"), 1000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("1", "1 kHz"), 1000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("2", "1 kHz"), 2000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("1 m", "1 kHz"), 1000*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("1 m", "60 rpm"), 1*2*math.pi)

    def test_rotating_liquid_pressure(self):
        self.assertAlmostEqual(rotating_liquid_pressure("1000 kg/m^3", "1 Hz", "1 m"), 1000 * (2 * math.pi) ** 2, places=6)
        self.assertAlmostEqual(rotating_liquid_pressure("1 g/cm^3", "60 rpm", "1 m"), 1000 * (2 * math.pi) ** 2, places=6)
        self.assertAlmostEqual(rotating_liquid_pressure("1000 g/L", "60 rpm", "100 cm"), 1000 * (2 * math.pi) ** 2, places=6)

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(LengthMeters)
        self.assertIsNotNone(FrequencyHz)
        self.assertIsNotNone(RotationRate)
        self.assertIsNotNone(MassGrams)
        self.assertIsNotNone(DensityKgPerM3)

    def test_rotation_functions_with_type_annotations(self):
        """Test rotation functions work with various unit inputs using new type annotations."""
        # Test rpm_to_Hz with various inputs
        self.assertAlmostEqual(rpm_to_Hz("60 rpm"), 1.)
        self.assertAlmostEqual(rpm_to_Hz("120 rpm"), 2.)

        # Test hz_to_rpm with various inputs
        self.assertAlmostEqual(hz_to_rpm("1 Hz"), 60.)
        self.assertAlmostEqual(hz_to_rpm("2 Hz"), 120.)

        # Test angular_speed with various inputs
        self.assertAlmostEqual(angular_speed("1 Hz"), 1*2*math.pi)
        self.assertAlmostEqual(angular_speed("60 rpm"), 1*2*math.pi)

        # Test rotation_linear_speed with various inputs
        self.assertAlmostEqual(rotation_linear_speed("1 m", "1 Hz"), 1*2*math.pi)
        self.assertAlmostEqual(rotation_linear_speed("1 m", "60 rpm"), 1*2*math.pi)

        # Test centrifugal_force with various inputs
        self.assertAlmostEqual(centrifugal_force("1 m", "60 rpm", "500 g"), 0.5 * (2 * math.pi) ** 2, places=6)
        self.assertAlmostEqual(centrifugal_force("1 m", "1 Hz", "0.5 kg"), 0.5 * (2 * math.pi) ** 2, places=6)

        # Test rotating_liquid_pressure with various inputs
        self.assertAlmostEqual(rotating_liquid_pressure("1 g/cm^3", "60 rpm", "1 m"), 1000 * (2 * math.pi) ** 2, places=6)
        self.assertAlmostEqual(rotating_liquid_pressure("1000 kg/m^3", "1 Hz", "1 m"), 1000 * (2 * math.pi) ** 2, places=6)

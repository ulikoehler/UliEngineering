#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.NoiseDensity import (
    actual_noise, noise_density,
    normalize_voltage, VoltageVolt,
    normalize_frequency, FrequencyHz
)
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

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(VoltageVolt)
        self.assertIsNotNone(FrequencyHz)

    def test_normalize_voltage_various_units(self):
        """Test normalize_voltage with various unit inputs."""
        test_cases = [
            ("1 V", 1.0),
            ("1 Volt", 1.0),
            ("1 volt", 1.0),
            ("1 mV", 1e-3),
            ("1 µV", 1e-6),
            ("1 nV", 1e-9),
            ("1 pV", 1e-12),
            ("1 kV", 1e3),
            ("1 MV", 1e6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_voltage(input_val)
                assert_approx_equal(result, expected)

    def test_noise_density_various_units(self):
        """Test noise density functions with various unit inputs."""
        # Test with different voltage units
        n1 = actual_noise("100 µV", "100 Hz")
        n2 = actual_noise("0.0001 V", "100 Hz")
        assert_approx_equal(n1, n2)

        # Test with different frequency units
        n3 = actual_noise("100 µV", "0.1 kHz")
        n4 = actual_noise("100 µV", "100 Hz")
        assert_approx_equal(n3, n4)

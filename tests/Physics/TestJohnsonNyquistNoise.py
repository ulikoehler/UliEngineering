#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.JohnsonNyquistNoise import (
    johnson_nyquist_noise_current, johnson_nyquist_noise_voltage,
    normalize_resistance, ResistanceOhm,
    normalize_temperature, TemperatureKelvin,
    normalize_frequency, FrequencyHz
)
from UliEngineering.EngineerIO import auto_format
import unittest

class TestJohnsonNyquistNoise(unittest.TestCase):
    def test_johnson_nyquist_noise_current(self):
        v = johnson_nyquist_noise_current("20 MΩ", "Δ10000 Hz", "20 °C")
        assert_approx_equal(v, 2.84512e-12, significant=5)
        self.assertEqual(auto_format(johnson_nyquist_noise_current, "20 MΩ", "Δ10000 Hz", "20 °C"), "2.85 pA")

    def test_johnson_nyquist_noise_voltage(self):
        v = johnson_nyquist_noise_voltage("20 MΩ", "Δ10000 Hz", "20 °C")
        self.assertEqual(auto_format(johnson_nyquist_noise_voltage, "20 MΩ", "Δ10000 Hz", "20 °C"), "56.9 µV")
        assert_approx_equal(v, 56.9025e-6, significant=5)

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(ResistanceOhm)
        self.assertIsNotNone(TemperatureKelvin)
        self.assertIsNotNone(FrequencyHz)

    def test_normalize_resistance_various_units(self):
        """Test normalize_resistance with various unit inputs"""
        test_cases = [
            ("1 Ω", 1.0),
            ("1 Ohm", 1.0),
            ("1 ohm", 1.0),
            ("1 kΩ", 1000.0),
            ("1 MΩ", 1e6),
            ("1 GΩ", 1e9),
            ("1 mΩ", 1e-3),
            ("1 µΩ", 1e-6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_resistance(input_val)
                assert_approx_equal(result, expected)

    def test_johnson_nyquist_noise_various_units(self):
        """Test Johnson Nyquist noise functions with various unit inputs"""
        # Test with different resistance units
        v1 = johnson_nyquist_noise_current("20000000 Ω", "10000 Hz", "20 °C")
        v2 = johnson_nyquist_noise_current("20 MΩ", "10000 Hz", "20 °C")
        assert_approx_equal(v1, v2)

        # Test with different frequency units
        v3 = johnson_nyquist_noise_current("20 MΩ", "10 kHz", "20 °C")
        v4 = johnson_nyquist_noise_current("20 MΩ", "10000 Hz", "20 °C")
        assert_approx_equal(v3, v4)

        # Test with different temperature units
        v5 = johnson_nyquist_noise_current("20 MΩ", "10000 Hz", "293.15 K")
        v6 = johnson_nyquist_noise_current("20 MΩ", "10000 Hz", "20 °C")
        assert_approx_equal(v5, v6)

        # Test voltage with different units
        v7 = johnson_nyquist_noise_voltage("20000000 Ω", "10000 Hz", "20 °C")
        v8 = johnson_nyquist_noise_voltage("20 MΩ", "10000 Hz", "20 °C")
        assert_approx_equal(v7, v8)

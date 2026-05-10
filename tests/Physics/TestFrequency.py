#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.Physics.Frequency import (
    frequency_to_period, normalize_frequency, normalize_rpm,
    FrequencyHz, RotationFrequency, RotationRate
)
import unittest

class TestFrequencies(unittest.TestCase):
    def test_frequency_to_period(self):
        assert_approx_equal(frequency_to_period(0.1), 10)
        assert_approx_equal(frequency_to_period("0.1 Hz"), 10)
        assert_approx_equal(frequency_to_period("10 Hz"), 0.1)
        assert_approx_equal(frequency_to_period("10 kHz"), 0.1e-3)

    def test_period_to_frequency(self):
        assert_approx_equal(frequency_to_period(10), 0.1)
        assert_approx_equal(frequency_to_period("10 s"), 0.1)
        assert_approx_equal(frequency_to_period("10 ks"), 0.1e-3)
        assert_approx_equal(frequency_to_period("1 ms"), 1e3)

    def test_normalize_frequency(self):
        # Test Hz normalization (SI unit)
        assert_approx_equal(normalize_frequency("1 Hz"), 1.0)
        assert_approx_equal(normalize_frequency("1 kHz"), 1000.0)
        assert_approx_equal(normalize_frequency("1 MHz"), 1e6)
        assert_approx_equal(normalize_frequency("1 GHz"), 1e9)
        # Test rpm to Hz conversion
        assert_approx_equal(normalize_frequency("60 rpm"), 1.0)
        assert_approx_equal(normalize_frequency("120 rpm"), 2.0)
        assert_approx_equal(normalize_frequency("3600 rpm"), 60.0)

    def test_normalize_rpm(self):
        # Test that normalize_rpm converts to Hz (SI unit)
        assert_approx_equal(normalize_rpm("60 rpm"), 1.0)
        assert_approx_equal(normalize_rpm("120 rpm"), 2.0)
        assert_approx_equal(normalize_rpm("3600 rpm"), 60.0)
        assert_approx_equal(normalize_rpm("1 rpm"), 1.0/60.0)

    def test_normalize_frequency_iterables(self):
        assert_allclose(normalize_frequency(["60 rpm", "120 rpm"]), [1.0, 2.0])
        assert_allclose(normalize_frequency(item for item in ["60 rpm", "120 rpm"]), [1.0, 2.0])
        assert_allclose(normalize_rpm(["60 rpm", "120 rpm"]), [1.0, 2.0])

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        # These should be importable and are Annotated types
        self.assertIsNotNone(FrequencyHz)
        self.assertIsNotNone(RotationFrequency)
        self.assertIsNotNone(RotationRate)
        # RotationFrequency should be an alias for FrequencyHz
        self.assertEqual(RotationFrequency, FrequencyHz)

    def test_frequency_type_various_units(self):
        """Test FrequencyHz type with various unit inputs."""
        test_cases = [
            ("1 Hz", 1.0),
            ("1 kHz", 1000.0),
            ("1 MHz", 1e6),
            ("60 rpm", 1.0),
            ("120 rpm", 2.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_frequency(input_val)
                assert_approx_equal(result, expected)

    def test_rotation_rate_type_various_units(self):
        """Test RotationRate type with various unit inputs."""
        test_cases = [
            ("60 rpm", 1.0),
            ("120 rpm", 2.0),
            ("3600 rpm", 60.0),
            ("1 rpm", 1.0/60.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_rpm(input_val)
                assert_approx_equal(result, expected)

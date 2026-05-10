#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.EngineerIO import auto_format
from UliEngineering.Physics.RF import (
    quality_factor, resonant_impedance, resonant_frequency, resonant_inductance,
    normalize_inductance, InductanceHenry,
    normalize_capacitance, CapacitanceFarad,
    normalize_frequency, FrequencyHz
)
import unittest

class TestRF(unittest.TestCase):
    def test_quality_factor(self):
        assert_approx_equal(quality_factor("8.000 MHz", "1 kHz"), 8000.0)
        assert_approx_equal(quality_factor("8.000 MHz", "1 MHz"), 8.0)

    def test_resonant_impedance(self):
        assert_approx_equal(resonant_impedance("100 uH", "10 nF", Q=30.0), 10./3)
        self.assertEqual(auto_format(resonant_impedance, "100 uH", "10 nF", Q=30.0), '3.33 Ω')

    def test_resonant_frequency(self):
        assert_approx_equal(resonant_frequency("100 uH", "10 nF"), 159154.94309189534)
        self.assertEqual(auto_format(resonant_frequency, "100 uH", "10 nF"), '159 kHz')

    def test_resonant_inductance(self):
        assert_approx_equal(resonant_inductance("250 kHz", "10 nF"), 4.052847345693511e-05)
        self.assertEqual(auto_format(resonant_inductance, "250 kHz", "10 nF"), '40.5 µH')

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(InductanceHenry)
        self.assertIsNotNone(CapacitanceFarad)
        self.assertIsNotNone(FrequencyHz)

    def test_normalize_inductance_various_units(self):
        """Test normalize_inductance with various unit inputs."""
        test_cases = [
            ("1 H", 1.0),
            ("1 Henry", 1.0),
            ("1 henry", 1.0),
            ("1 mH", 1e-3),
            ("1 µH", 1e-6),
            ("1 nH", 1e-9),
            ("1 pH", 1e-12),
            ("1 kH", 1e3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_inductance(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_capacitance_various_units(self):
        """Test normalize_capacitance with various unit inputs."""
        test_cases = [
            ("1 F", 1.0),
            ("1 Farad", 1.0),
            ("1 farad", 1.0),
            ("1 mF", 1e-3),
            ("1 µF", 1e-6),
            ("1 nF", 1e-9),
            ("1 pF", 1e-12),
            ("1 kF", 1e3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_capacitance(input_val)
                assert_approx_equal(result, expected)

    def test_rf_functions_various_units(self):
        """Test RF functions with various unit inputs."""
        # Test quality_factor with different frequency units
        q1 = quality_factor("8 MHz", "1 kHz")
        q2 = quality_factor("8000 kHz", "1000 Hz")
        assert_approx_equal(q1, q2)

        # Test resonant_impedance with different inductance units
        i1 = resonant_impedance("100 µH", "10 nF", Q=30.0)
        i2 = resonant_impedance("0.0001 H", "10 nF", Q=30.0)
        assert_approx_equal(i1, i2)

        # Test resonant_frequency with different capacitance units
        f1 = resonant_frequency("100 µH", "10 nF")
        f2 = resonant_frequency("100 µH", "0.00000001 F")
        assert_approx_equal(f1, f2)

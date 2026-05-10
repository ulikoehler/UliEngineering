#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.MagneticResonance import (
    larmor_frequency, NucleusLarmorFrequency,
    normalize_magnetic_field, MagneticFieldTesla,
    FrequencyHz
)
import unittest

class TestLarmorFrequency(unittest.TestCase):
    def test_larmor_frequency_h1(self):
        self.assertAlmostEqual(larmor_frequency(0., nucleus_larmor_frequency=NucleusLarmorFrequency.H1), 0)
        self.assertAlmostEqual(larmor_frequency(1., nucleus_larmor_frequency=NucleusLarmorFrequency.H1), 42.57638543e6, places=6)
        self.assertAlmostEqual(larmor_frequency(2.2, nucleus_larmor_frequency=NucleusLarmorFrequency.H1), 2.2*42.57638543e6)
        # H1 should be the standard value
        self.assertAlmostEqual(larmor_frequency(0.), 0)

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(MagneticFieldTesla)
        self.assertIsNotNone(FrequencyHz)

    def test_normalize_magnetic_field_various_units(self):
        """Test normalize_magnetic_field with various unit inputs"""
        test_cases = [
            ("1 T", 1.0),
            ("1 Tesla", 1.0),
            ("1 tesla", 1.0),
            ("1 mT", 1e-3),
            ("1 µT", 1e-6),
            ("1 G", 1e-4),
            ("1 Gauss", 1e-4),
            ("1 gauss", 1e-4),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_magnetic_field(input_val)
                assert_approx_equal(result, expected)

    def test_larmor_frequency_various_units(self):
        """Test larmor_frequency with various magnetic field units"""
        # Test with different magnetic field units
        f1 = larmor_frequency("1 T", nucleus_larmor_frequency=NucleusLarmorFrequency.H1)
        f2 = larmor_frequency("1 Tesla", nucleus_larmor_frequency=NucleusLarmorFrequency.H1)
        assert_approx_equal(f1, f2)

        # Test with mT
        f3 = larmor_frequency("1000 mT", nucleus_larmor_frequency=NucleusLarmorFrequency.H1)
        f4 = larmor_frequency("1 T", nucleus_larmor_frequency=NucleusLarmorFrequency.H1)
        assert_approx_equal(f3, f4)

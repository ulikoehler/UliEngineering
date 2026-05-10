#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.Physics.Mass import normalize_mass_grams, MassGrams
import unittest


class TestMassNormalization(unittest.TestCase):
    def test_normalize_mass_grams(self):
        # Test gram normalization (SI unit for this function)
        assert_approx_equal(normalize_mass_grams("1 g"), 1.0)
        assert_approx_equal(normalize_mass_grams("500 g"), 500.0)
        assert_approx_equal(normalize_mass_grams("1000 g"), 1000.0)
        # Test kg to grams conversion
        assert_approx_equal(normalize_mass_grams("1 kg"), 1000.0)
        assert_approx_equal(normalize_mass_grams("0.5 kg"), 500.0)
        assert_approx_equal(normalize_mass_grams("2.5 kg"), 2500.0)
        # Test mg to grams conversion
        assert_approx_equal(normalize_mass_grams("1000 mg"), 1.0)
        assert_approx_equal(normalize_mass_grams("500 mg"), 0.5)
        assert_approx_equal(normalize_mass_grams("1 mg"), 0.001)

    def test_normalize_mass_iterables(self):
        assert_allclose(normalize_mass_grams(["500 g", "0.5 kg"]), [500.0, 500.0])
        assert_allclose(normalize_mass_grams(["1 kg", "1000 mg", "500 g"]), [1000.0, 1.0, 500.0])

    def test_type_annotation_exists(self):
        """Test that the new type annotation is available."""
        self.assertIsNotNone(MassGrams)

    def test_mass_type_various_units(self):
        """Test MassGrams type with various unit inputs."""
        test_cases = [
            ("1 g", 1.0),
            ("500 g", 500.0),
            ("1 kg", 1000.0),
            ("0.5 kg", 500.0),
            ("1000 mg", 1.0),
            ("500 mg", 0.5),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_mass_grams(input_val)
                assert_approx_equal(result, expected)
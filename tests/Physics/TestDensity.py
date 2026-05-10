#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.Physics.Density import normalize_density_kg_per_m3, DensityKgPerM3
import unittest


class TestDensityNormalization(unittest.TestCase):
    def test_normalize_density(self):
        # Test kg/m^3 normalization (SI unit)
        assert_approx_equal(normalize_density_kg_per_m3("1 kg/m^3"), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3("1000 kg/m^3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 kg/m3"), 1.0)
        # Test g/cm^3 to kg/m^3 conversion
        assert_approx_equal(normalize_density_kg_per_m3("1 g/cm^3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("2 g/cm^3"), 2000.0)
        assert_approx_equal(normalize_density_kg_per_m3("2.7 g/cm^3"), 2700.0)
        # Test g/cm3 to kg/m^3 conversion
        assert_approx_equal(normalize_density_kg_per_m3("1 g/cm3"), 1000.0)
        # Test g/L to kg/m^3 conversion
        assert_approx_equal(normalize_density_kg_per_m3("1 g/L"), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3("1000 g/L"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 g/l"), 1.0)

    def test_normalize_density_iterables(self):
        assert_allclose(normalize_density_kg_per_m3(["1 g/cm^3", "2 g/cm^3"]), [1000.0, 2000.0])
        assert_allclose(normalize_density_kg_per_m3(["1 kg/m^3", "2 g/L"]), [1.0, 2.0])

    def test_type_annotation_exists(self):
        """Test that the new type annotation is available."""
        self.assertIsNotNone(DensityKgPerM3)

    def test_density_type_various_units(self):
        """Test DensityKgPerM3 type with various unit inputs."""
        test_cases = [
            ("1 kg/m^3", 1.0),
            ("1 kg/m3", 1.0),
            ("1000 kg/m^3", 1000.0),
            ("1 g/cm^3", 1000.0),
            ("2 g/cm^3", 2000.0),
            ("1 g/cm3", 1000.0),
            ("1 g/L", 1.0),
            ("1000 g/L", 1000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_density_kg_per_m3(input_val)
                assert_approx_equal(result, expected)
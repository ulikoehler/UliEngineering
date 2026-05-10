#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.Pressure import (
    pascal_to_bar, bar_to_pascal, barlow_tangential,
    normalize_pressure_pascal, normalize_pressure_bar,
    PressurePascal, PressureBar,
    psi_to_pascal, psi_to_bar, pascal_to_psi, bar_to_psi
)
import unittest

class TestPressureConversion(unittest.TestCase):
    def test_pascal_to_bar(self):
        self.assertAlmostEqual(pascal_to_bar(0.), 0)
        self.assertAlmostEqual(pascal_to_bar(1.), 1/100000)
        self.assertAlmostEqual(pascal_to_bar(5.), 5/100000)
        self.assertAlmostEqual(pascal_to_bar(100000), 1)

    def test_bar_to_pascal(self):
        self.assertAlmostEqual(bar_to_pascal(0.), 0)
        self.assertAlmostEqual(bar_to_pascal(1.), 100000)
        self.assertAlmostEqual(bar_to_pascal(5.), 500000)
        self.assertAlmostEqual(bar_to_pascal(0.00001), 1)

    def test_normalize_pressure_pascal(self):
        # Test Pa normalization (SI unit)
        assert_approx_equal(normalize_pressure_pascal("1 Pa"), 1.0)
        assert_approx_equal(normalize_pressure_pascal("1000 Pa"), 1000.0)
        # Test bar to Pa conversion
        assert_approx_equal(normalize_pressure_pascal("1 bar"), 1e5)
        assert_approx_equal(normalize_pressure_pascal("2 bar"), 2e5)
        assert_approx_equal(normalize_pressure_pascal("0.5 bar"), 5e4)
        # Test psi to Pa conversion
        assert_approx_equal(normalize_pressure_pascal("1 psi"), 6894.76)
        assert_approx_equal(normalize_pressure_pascal("10 psi"), 68947.6)

    def test_normalize_pressure_bar(self):
        # Test bar normalization (SI unit for this function)
        assert_approx_equal(normalize_pressure_bar("1 bar"), 1.0)
        assert_approx_equal(normalize_pressure_bar("2 bar"), 2.0)
        # Test Pa to bar conversion
        assert_approx_equal(normalize_pressure_bar("1e5 Pa"), 1.0)
        assert_approx_equal(normalize_pressure_bar("2e5 Pa"), 2.0)
        assert_approx_equal(normalize_pressure_bar("1 Pa"), 1e-5)
        # Test psi to bar conversion
        assert_approx_equal(normalize_pressure_bar("1 psi"), 0.0689476)
        assert_approx_equal(normalize_pressure_bar("10 psi"), 0.689476)

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(PressurePascal)
        self.assertIsNotNone(PressureBar)

    def test_pressure_pascal_various_units(self):
        """Test PressurePascal type with various unit inputs."""
        test_cases = [
            ("1 Pa", 1.0),
            ("1000 Pa", 1000.0),
            ("1 bar", 1e5),
            ("2 bar", 2e5),
            ("1 psi", 6894.76),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_pressure_pascal(input_val)
                assert_approx_equal(result, expected)

    def test_pressure_bar_various_units(self):
        """Test PressureBar type with various unit inputs."""
        test_cases = [
            ("1 bar", 1.0),
            ("2 bar", 2.0),
            ("1e5 Pa", 1.0),
            ("2e5 Pa", 2.0),
            ("1 Pa", 1e-5),
            ("1 psi", 0.0689476),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_pressure_bar(input_val)
                assert_approx_equal(result, expected)

    def test_barlow_tangential(self):
        """Test barlow_tangential function with new type annotations."""
        # Test with numeric inputs
        result = barlow_tangential(0.1, 0.08, 1e5)
        self.assertAlmostEqual(result, 450000.0)

    def test_psi_to_pascal(self):
        """Test psi to Pascal conversion."""
        self.assertAlmostEqual(psi_to_pascal(0.), 0)
        self.assertAlmostEqual(psi_to_pascal(1.), 6894.76)
        self.assertAlmostEqual(psi_to_pascal(10.), 68947.6)
        self.assertAlmostEqual(psi_to_pascal(100.), 689476.0)

    def test_psi_to_bar(self):
        """Test psi to bar conversion."""
        self.assertAlmostEqual(psi_to_bar(0.), 0)
        self.assertAlmostEqual(psi_to_bar(1.), 0.0689476)
        self.assertAlmostEqual(psi_to_bar(10.), 0.689476)
        self.assertAlmostEqual(psi_to_bar(100.), 6.89476)

    def test_pascal_to_psi(self):
        """Test Pascal to psi conversion."""
        self.assertAlmostEqual(pascal_to_psi(0.), 0)
        self.assertAlmostEqual(pascal_to_psi(6894.76), 1.0, places=5)
        self.assertAlmostEqual(pascal_to_psi(68947.6), 10.0, places=5)
        self.assertAlmostEqual(pascal_to_psi(689476.0), 100.0, places=5)

    def test_bar_to_psi(self):
        """Test bar to psi conversion."""
        self.assertAlmostEqual(bar_to_psi(0.), 0)
        self.assertAlmostEqual(bar_to_psi(0.0689476), 1.0)
        self.assertAlmostEqual(bar_to_psi(0.689476), 10.0)
        self.assertAlmostEqual(bar_to_psi(6.89476), 100.0)

    def test_psi_conversions_roundtrip(self):
        """Test that psi conversions are reversible."""
        # psi -> Pascal -> psi
        psi = 10.0
        pa = psi_to_pascal(psi)
        psi_back = pascal_to_psi(pa)
        self.assertAlmostEqual(psi, psi_back)

        # psi -> bar -> psi
        bar = psi_to_bar(psi)
        psi_back = bar_to_psi(bar)
        self.assertAlmostEqual(psi, psi_back)

        # bar -> Pascal -> bar (existing test)
        bar = 1.0
        pa = bar_to_pascal(bar)
        bar_back = pascal_to_bar(pa)
        self.assertAlmostEqual(bar, bar_back)
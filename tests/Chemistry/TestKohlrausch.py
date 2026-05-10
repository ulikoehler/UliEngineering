#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.Kohlrausch import (
    kohlrausch_limiting_molar_conductivity,
    kohlrausch_molar_conductivity,
    kohlrausch_coefficient_from_data,
    transference_number,
    LIMITING_MOLAR_CONDUCTIVITIES,
    normalize_molar_conductivity, MolarConductivitySCm2Mol,
    normalize_concentration, ConcentrationMolar,
)


class TestKohlrausch(unittest.TestCase):
    def test_kohlrausch_limiting_molar_conductivity_scalar(self):
        """Test limiting molar conductivity with scalar input"""
        Lambda_0 = kohlrausch_limiting_molar_conductivity([50.10, 76.35], [1, 1])
        self.assertIsInstance(Lambda_0, float)
        self.assertAlmostEqual(Lambda_0, 126.45, places=2)

    def test_kohlrausch_molar_conductivity_scalar(self):
        """Test molar conductivity with scalar input"""
        Lambda_m = kohlrausch_molar_conductivity(Lambda_0=126.45, K=10.0, c=0.1)
        self.assertIsInstance(Lambda_m, float)
        self.assertLess(Lambda_m, 126.45)

    def test_kohlrausch_coefficient_from_data_scalar(self):
        """Test Kohlrausch coefficient from data with scalar input"""
        K = kohlrausch_coefficient_from_data(Lambda_0=126.45, Lambda_m=120.0, c=0.1)
        self.assertIsInstance(K, float)
        self.assertGreater(K, 0)

    def test_transference_number_scalar(self):
        """Test transference number with scalar input"""
        t = transference_number(lambda_ion=50.10, Lambda_0=126.45)
        self.assertIsInstance(t, float)
        self.assertGreater(t, 0)
        self.assertLess(t, 1)

    def test_limiting_molar_conductivities_dict(self):
        """Test that the limiting molar conductivities dictionary exists"""
        self.assertIsInstance(LIMITING_MOLAR_CONDUCTIVITIES, dict)
        self.assertIn("Na+", LIMITING_MOLAR_CONDUCTIVITIES)
        self.assertIn("Cl-", LIMITING_MOLAR_CONDUCTIVITIES)

    def test_inverse_kohlrausch_equation(self):
        """Test inverse relationship between molar conductivity functions"""
        Lambda_0 = 126.45
        c = 0.1
        K = 10.0
        Lambda_m = kohlrausch_molar_conductivity(Lambda_0, K, c)
        K_calc = kohlrausch_coefficient_from_data(Lambda_0, Lambda_m, c)
        self.assertAlmostEqual(K, K_calc, places=10)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(MolarConductivitySCm2Mol)
        self.assertIsNotNone(ConcentrationMolar)

    def test_normalize_molar_conductivity_various_units(self):
        """Test normalize_molar_conductivity with various unit inputs"""
        test_cases = [
            ("1 S·cm²/mol", 1.0),
            ("1 S·m²/mol", 10000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_molar_conductivity(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_concentration_various_units(self):
        """Test normalize_concentration with various unit inputs"""
        test_cases = [
            ("1 mol/L", 1.0),
            ("1 M", 1.0),
            ("1 mM", 1e-3),
            ("1 µM", 1e-6),
            ("1 mol/m³", 1e-3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_concentration(input_val)
                self.assertAlmostEqual(result, expected)

    def test_kohlrausch_functions_various_units(self):
        """Test Kohlrausch functions with various unit inputs"""
        # Test with different concentration units
        K1 = kohlrausch_molar_conductivity("126.45 S·cm²/mol", 10.0, "0.1 M")
        K2 = kohlrausch_molar_conductivity("126.45 S·cm²/mol", 10.0, "100 mM")
        self.assertAlmostEqual(K1, K2)


if __name__ == "__main__":
    unittest.main()

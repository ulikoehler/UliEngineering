#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.Langmuir import (
    langmuir_coverage,
    langmuir_adsorbed_amount,
    langmuir_constant_from_coverage,
    langmuir_competitive_coverage,
    langmuir_inverse_linearized,
    langmuir_dissociation_rate,
    normalize_concentration, ConcentrationMolar,
    normalize_adsorption_capacity, AdsorptionCapacityMolG,
    normalize_langmuir_constant, LangmuirConstantLMol,
    normalize_rate_constant, RateConstantLMolS,
)


class TestLangmuir(unittest.TestCase):
    def test_langmuir_coverage_scalar(self):
        """Test Langmuir coverage with scalar input"""
        theta = langmuir_coverage(K=1.0, C=0.5)
        self.assertIsInstance(theta, float)
        self.assertGreater(theta, 0)
        self.assertLess(theta, 1)

    def test_langmuir_adsorbed_amount_scalar(self):
        """Test Langmuir adsorbed amount with scalar input."""
        q = langmuir_adsorbed_amount(q_max=1.0, K=1.0, C=0.5)
        self.assertIsInstance(q, float)
        self.assertGreater(q, 0)

    def test_langmuir_constant_from_coverage_scalar(self):
        """Test Langmuir constant from coverage with scalar input"""
        K = langmuir_constant_from_coverage(theta=0.5, C=1.0)
        self.assertIsInstance(K, float)
        self.assertGreater(K, 0)

    def test_langmuir_competitive_coverage_scalar(self):
        """Test Langmuir competitive coverage with scalar input"""
        theta = langmuir_competitive_coverage(K_i=1.0, C_i=0.5, K_all=[1.0, 0.5], C_all=[0.5, 0.5])
        self.assertIsInstance(theta, float)
        self.assertGreater(theta, 0)

    def test_langmuir_inverse_linearized_scalar(self):
        """Test Langmuir inverse linearized with scalar input"""
        inv_c, inv_q = langmuir_inverse_linearized(C=0.5, q=0.25)
        self.assertIsInstance(inv_c, float)
        self.assertIsInstance(inv_q, float)

    def test_langmuir_dissociation_rate_scalar(self):
        """Test Langmuir dissociation rate with scalar input"""
        k_des = langmuir_dissociation_rate(k_ads=0.1, K=1.0)
        self.assertIsInstance(k_des, float)
        self.assertGreater(k_des, 0)

    def test_inverse_langmuir_coverage(self):
        """Test inverse relationship between coverage and constant functions"""
        K = 1.0
        C = 0.5
        theta = langmuir_coverage(K, C)
        K_calc = langmuir_constant_from_coverage(theta, C)
        self.assertAlmostEqual(K, K_calc, places=10)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(ConcentrationMolar)
        self.assertIsNotNone(AdsorptionCapacityMolG)
        self.assertIsNotNone(LangmuirConstantLMol)
        self.assertIsNotNone(RateConstantLMolS)

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

    def test_normalize_adsorption_capacity_various_units(self):
        """Test normalize_adsorption_capacity with various unit inputs"""
        test_cases = [
            ("1 mol/g", 1.0),
            ("1 mmol/g", 1e-3),
            ("1 µmol/g", 1e-6),
            ("1 mol/kg", 1e-3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_adsorption_capacity(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_langmuir_constant_various_units(self):
        """Test normalize_langmuir_constant with various unit inputs"""
        test_cases = [
            ("1 L/mol", 1.0),
            ("1 m³/mol", 1000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_langmuir_constant(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_rate_constant_various_units(self):
        """Test normalize_rate_constant with various unit inputs"""
        test_cases = [
            ("1 L/(mol·s)", 1.0),
            ("1 m³/(mol·s)", 1000.0),
            ("1 1/s", 1.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_rate_constant(input_val)
                self.assertAlmostEqual(result, expected)

    def test_langmuir_functions_various_units(self):
        """Test Langmuir functions with various unit inputs"""
        # Test with different concentration units
        theta1 = langmuir_coverage("1 L/mol", "0.5 M")
        theta2 = langmuir_coverage("1 L/mol", "500 mM")
        self.assertAlmostEqual(theta1, theta2)


if __name__ == "__main__":
    unittest.main()

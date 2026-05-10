#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.DebyeHuckel import (
    debye_huckel_limiting_law,
    debye_huckel_extended,
    debye_huckel_activity_coefficient,
    debye_huckel_extended_activity_coefficient,
    debye_length,
    normalize_ionic_strength, IonicStrengthMolar,
    normalize_ion_diameter, IonDiameterNm,
)
import numpy as np


class TestDebyeHuckel(unittest.TestCase):
    def test_debye_huckel_limiting_law_scalar(self):
        """Test Debye-Hückel limiting law with scalar input"""
        log_gamma = debye_huckel_limiting_law(z_plus=1, z_minus=1, I=0.01)
        self.assertIsInstance(log_gamma, float)

    def test_debye_huckel_limiting_law_array(self):
        """Test Debye-Hückel limiting law with array input."""
        I = np.array([0.001, 0.01, 0.1])
        log_gamma = debye_huckel_limiting_law(z_plus=1, z_minus=1, I=I)
        self.assertEqual(len(log_gamma), 3)

    def test_debye_huckel_extended_scalar(self):
        """Test extended Debye-Hückel equation with scalar input."""
        log_gamma = debye_huckel_extended(z=1, I=0.1, a=0.3)
        self.assertIsInstance(log_gamma, float)

    def test_debye_huckel_extended_array(self):
        """Test extended Debye-Hückel equation with array input"""
        I = np.array([0.01, 0.1, 0.5])
        log_gamma = debye_huckel_extended(z=1, I=I, a=0.3)
        self.assertEqual(len(log_gamma), 3)

    def test_debye_huckel_activity_coefficient_scalar(self):
        """Test activity coefficient with scalar input"""
        gamma = debye_huckel_activity_coefficient(z_plus=1, z_minus=1, I=0.01)
        self.assertIsInstance(gamma, float)
        self.assertLess(gamma, 1.0)

    def test_debye_huckel_extended_activity_coefficient_scalar(self):
        """Test extended activity coefficient with scalar input"""
        gamma = debye_huckel_extended_activity_coefficient(z=1, I=0.1, a=0.3)
        self.assertIsInstance(gamma, float)
        self.assertLess(gamma, 1.0)

    def test_consistency_limiting_law(self):
        """Test consistency between log and linear forms for limiting law"""
        z_plus = 1
        z_minus = 1
        I = 0.01
        log_gamma = debye_huckel_limiting_law(z_plus, z_minus, I)
        gamma = debye_huckel_activity_coefficient(z_plus, z_minus, I)
        self.assertAlmostEqual(gamma, 10.0 ** log_gamma, places=10)

    def test_consistency_extended(self):
        """Test consistency between log and linear forms for extended equation."""
        z = 1
        I = 0.1
        a = 0.3
        log_gamma = debye_huckel_extended(z, I, a)
        gamma = debye_huckel_extended_activity_coefficient(z, I, a)
        self.assertAlmostEqual(gamma, 10.0 ** log_gamma, places=10)

    def test_debye_length_scalar(self):
        """Test Debye length with scalar input"""
        length = debye_length(I=0.1)
        self.assertIsInstance(length, float)
        self.assertGreater(length, 0)

    def test_debye_length_array(self):
        """Test Debye length with array input."""
        I = np.array([0.01, 0.1, 1.0])
        length = debye_length(I=I)
        self.assertEqual(len(length), 3)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(IonicStrengthMolar)
        self.assertIsNotNone(IonDiameterNm)

    def test_normalize_ionic_strength_various_units(self):
        """Test normalize_ionic_strength with various unit inputs"""
        test_cases = [
            ("1 M", 1.0),
            ("1 mol/L", 1.0),
            ("1 mol/l", 1.0),
            ("1 mM", 1e-3),
            ("1 µM", 1e-6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_ionic_strength(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_ion_diameter_various_units(self):
        """Test normalize_ion_diameter with various unit inputs."""
        test_cases = [
            ("1 nm", 1.0),
            ("1 pm", 1e-3),
            ("1 Å", 0.1),
            ("1e-9 m", 1.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_ion_diameter(input_val)
                self.assertAlmostEqual(result, expected)

    def test_debye_huckel_functions_various_units(self):
        """Test Debye-Hückel functions with various unit inputs"""
        # Test with different ionic strength units
        log1 = debye_huckel_limiting_law(1, 1, "0.01 M")
        log2 = debye_huckel_limiting_law(1, 1, "10 mM")
        self.assertAlmostEqual(log1, log2)


if __name__ == "__main__":
    unittest.main()

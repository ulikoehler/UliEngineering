#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.DaviesEquation import (
    davies_activity_coefficient,
    davies_log_activity_coefficient,
    debye_huckel_A_parameter,
    normalize_ionic_strength, IonicStrengthMolar,
)
import numpy as np


class TestDaviesEquation(unittest.TestCase):
    def test_debye_huckel_A_parameter_default(self):
        """Test Debye-Hückel A parameter at 25°C in water."""
        A = debye_huckel_A_parameter()
        self.assertAlmostEqual(A, 0.511, places=3)

    def test_debye_huckel_A_parameter_custom_temp(self):
        """Test Debye-Hückel A parameter at different temperature."""
        A = debye_huckel_A_parameter(T=300.0)
        self.assertAlmostEqual(A, 0.506, places=3)

    def test_davies_log_activity_coefficient_scalar(self):
        """Test Davies equation for log activity coefficient with scalar input."""
        log_gamma = davies_log_activity_coefficient(z=1, ionic_strength=0.1)
        self.assertIsInstance(log_gamma, float)

    def test_davies_log_activity_coefficient_array(self):
        """Test Davies equation for log activity coefficient with array input."""
        ionic_strength = np.array([0.01, 0.1, 0.5])
        log_gamma = davies_log_activity_coefficient(z=1, ionic_strength=ionic_strength)
        self.assertEqual(len(log_gamma), 3)

    def test_davies_activity_coefficient_scalar(self):
        """Test Davies equation for activity coefficient with scalar input."""
        gamma = davies_activity_coefficient(z=1, ionic_strength=0.1)
        self.assertIsInstance(gamma, float)
        self.assertLess(gamma, 1.0)  # Activity coefficient should be < 1 for ions

    def test_davies_activity_coefficient_array(self):
        """Test Davies equation for activity coefficient with array input."""
        ionic_strength = np.array([0.01, 0.1, 0.5])
        gamma = davies_activity_coefficient(z=1, ionic_strength=ionic_strength)
        self.assertEqual(len(gamma), 3)

    def test_davies_activity_coefficient_charge(self):
        """Test that higher charge gives lower activity coefficient."""
        gamma_z1 = davies_activity_coefficient(z=1, ionic_strength=0.1)
        gamma_z2 = davies_activity_coefficient(z=2, ionic_strength=0.1)
        self.assertLess(gamma_z2, gamma_z1)

    def test_davies_activity_coefficient_consistency(self):
        """Test consistency between log and linear forms."""
        z = 1
        ionic_strength = 0.1
        log_gamma = davies_log_activity_coefficient(z, ionic_strength)
        gamma = davies_activity_coefficient(z, ionic_strength)
        self.assertAlmostEqual(gamma, 10.0 ** log_gamma, places=10)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(IonicStrengthMolar)

    def test_normalize_ionic_strength_various_units(self):
        """Test normalize_ionic_strength with various unit inputs."""
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

    def test_davies_functions_various_units(self):
        """Test Davies functions with various unit inputs."""
        # Test davies_log_activity_coefficient with different units
        log1 = davies_log_activity_coefficient(1, "0.1 M")
        log2 = davies_log_activity_coefficient(1, "100 mM")
        self.assertAlmostEqual(log1, log2)


if __name__ == "__main__":
    unittest.main()

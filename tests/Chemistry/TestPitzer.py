#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.Pitzer import (
    pitzer_f_gamma,
    pitzer_B_gamma,
    pitzer_C_gamma,
    pitzer_activity_coefficient,
    pitzer_osmotic_coefficient,
    PITZER_A_PHI_25C,
    PITZER_PARAMETERS,
    normalize_molality, MolalityMolKg,
)


class TestPitzer(unittest.TestCase):
    def test_pitzer_f_gamma_scalar(self):
        """Test Pitzer f^γ with scalar input"""
        result = pitzer_f_gamma(I=1.0)
        self.assertIsInstance(result, float)

    def test_pitzer_B_gamma_scalar(self):
        """Test Pitzer B^γ with scalar input"""
        result = pitzer_B_gamma(I=1.0, beta0=0.0765, beta1=0.2664)
        self.assertIsInstance(result, float)

    def test_pitzer_C_gamma_scalar(self):
        """Test Pitzer C^γ with scalar input."""
        result = pitzer_C_gamma(C_phi=0.00127)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1.5 * 0.00127, places=6)

    def test_pitzer_activity_coefficient_scalar(self):
        """Test Pitzer activity coefficient with scalar input."""
        beta0, beta1, C_phi = PITZER_PARAMETERS["NaCl"]
        result = pitzer_activity_coefficient(m=1.0, z_plus=1, z_minus=1, nu_plus=1, nu_minus=1,
                                              beta0=beta0, beta1=beta1, C_phi=C_phi)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_pitzer_osmotic_coefficient_scalar(self):
        """Test Pitzer osmotic coefficient with scalar input."""
        beta0, beta1, C_phi = PITZER_PARAMETERS["NaCl"]
        result = pitzer_osmotic_coefficient(m=1.0, z_plus=1, z_minus=1, nu_plus=1, nu_minus=1,
                                             beta0=beta0, beta1=beta1, C_phi=C_phi)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_pitzer_parameters_dict(self):
        """Test that the Pitzer parameters dictionary exists."""
        self.assertIsInstance(PITZER_PARAMETERS, dict)
        self.assertIn("NaCl", PITZER_PARAMETERS)
        self.assertIn("KCl", PITZER_PARAMETERS)

    def test_pitzer_activity_coefficient_with_parameters(self):
        """Test activity coefficient using common electrolyte parameters."""
        beta0, beta1, C_phi = PITZER_PARAMETERS["NaCl"]
        gamma = pitzer_activity_coefficient(m=1.0, z_plus=1, z_minus=1, nu_plus=1, nu_minus=1,
                                            beta0=beta0, beta1=beta1, C_phi=C_phi)
        self.assertIsInstance(gamma, float)
        self.assertGreater(gamma, 0)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(MolalityMolKg)

    def test_normalize_molality_various_units(self):
        """Test normalize_molality with various unit inputs."""
        test_cases = [
            ("1 mol/kg", 1.0),
            ("1 m", 1.0),
            ("1 mmol/kg", 1e-3),
            ("1 µmol/kg", 1e-6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_molality(input_val)
                self.assertAlmostEqual(result, expected)

    def test_pitzer_functions_various_units(self):
        """Test Pitzer functions with various unit inputs."""
        # Test with different molality units
        f1 = pitzer_f_gamma("1 mol/kg")
        f2 = pitzer_f_gamma("1000 mmol/kg")
        self.assertAlmostEqual(f1, f2)


if __name__ == "__main__":
    unittest.main()

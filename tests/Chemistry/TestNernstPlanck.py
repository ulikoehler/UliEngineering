#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.NernstPlanck import (
    nernst_planck_flux,
    nernst_planck_diffusion_flux,
    nernst_planck_migration_flux,
    einstein_relation_diffusion_mobility,
    ionic_mobility_from_diffusion,
    normalize_diffusion_coefficient, DiffusionCoefficientM2S,
    normalize_concentration, ConcentrationMolM3,
    normalize_ionic_mobility, IonicMobilityM2VS,
)


class TestNernstPlanck(unittest.TestCase):
    def test_nernst_planck_flux_scalar(self):
        """Test Nernst-Planck flux with scalar input."""
        J = nernst_planck_flux(D=1e-9, dC_dx=1.0, z=1, C=1.0, dPhi_dx=1000.0)
        self.assertIsInstance(J, float)

    def test_nernst_planck_diffusion_flux_scalar(self):
        """Test Nernst-Planck diffusion flux with scalar input."""
        J = nernst_planck_diffusion_flux(D=1e-9, dC_dx=1.0)
        self.assertIsInstance(J, float)
        self.assertLess(J, 0)

    def test_nernst_planck_migration_flux_scalar(self):
        """Test Nernst-Planck migration flux with scalar input."""
        J = nernst_planck_migration_flux(D=1e-9, z=1, C=1.0, dPhi_dx=1000.0)
        self.assertIsInstance(J, float)

    def test_einstein_relation_diffusion_mobility_scalar(self):
        """Test Einstein relation diffusion mobility with scalar input."""
        D = einstein_relation_diffusion_mobility(mobility=1e-8)
        self.assertIsInstance(D, float)
        self.assertGreater(D, 0)

    def test_ionic_mobility_from_diffusion_scalar(self):
        """Test ionic mobility from diffusion with scalar input."""
        u = ionic_mobility_from_diffusion(D=1e-9, z=1)
        self.assertIsInstance(u, float)
        self.assertGreater(u, 0)

    def test_inverse_einstein_relation(self):
        """Test inverse relationship between Einstein relation functions."""
        D_orig = 1e-9
        u = ionic_mobility_from_diffusion(D_orig, 1)
        D_back = einstein_relation_diffusion_mobility(u)
        self.assertAlmostEqual(D_orig, D_back, places=10)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(DiffusionCoefficientM2S)
        self.assertIsNotNone(ConcentrationMolM3)
        self.assertIsNotNone(IonicMobilityM2VS)

    def test_normalize_diffusion_coefficient_various_units(self):
        """Test normalize_diffusion_coefficient with various unit inputs."""
        test_cases = [
            ("1 m²/s", 1.0),
            ("1 cm²/s", 1e-4),
            ("1 mm²/s", 1e-6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_diffusion_coefficient(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_concentration_various_units(self):
        """Test normalize_concentration with various unit inputs."""
        test_cases = [
            ("1 mol/m³", 1.0),
            ("1 mol/L", 1000.0),
            ("1 M", 1000.0),
            ("1 mM", 1.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_concentration(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_ionic_mobility_various_units(self):
        """Test normalize_ionic_mobility with various unit inputs."""
        test_cases = [
            ("1 m²/(V·s)", 1.0),
            ("1 cm²/(V·s)", 1e-4),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_ionic_mobility(input_val)
                self.assertAlmostEqual(result, expected)

    def test_nernst_planck_functions_various_units(self):
        """Test Nernst-Planck functions with various unit inputs."""
        # Test with different concentration units
        J1 = nernst_planck_flux("1e-9 m²/s", 1.0, 1, "1 mol/m³", 1000.0)
        J2 = nernst_planck_flux("1e-9 m²/s", 1.0, 1, "0.001 mol/L", 1000.0)
        self.assertAlmostEqual(J1, J2)


if __name__ == "__main__":
    unittest.main()

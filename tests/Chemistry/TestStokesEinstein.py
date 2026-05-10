#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.StokesEinstein import (
    stokes_einstein_diffusion,
    stokes_einstein_radius,
    stokes_einstein_viscosity,
    stokes_einstein_rotational_diffusion,
    WATER_VISCOSITY_25C,
    normalize_radius, RadiusM,
    normalize_viscosity, ViscosityPaS,
    normalize_diffusion_coefficient, DiffusionCoefficientM2S,
)


class TestStokesEinstein(unittest.TestCase):
    def test_stokes_einstein_diffusion_scalar(self):
        """Test Stokes-Einstein diffusion coefficient with scalar input."""
        D = stokes_einstein_diffusion(r=1e-9, eta=WATER_VISCOSITY_25C, T=298.15)
        self.assertIsInstance(D, float)
        self.assertGreater(D, 0)

    def test_stokes_einstein_radius_scalar(self):
        """Test Stokes-Einstein radius with scalar input."""
        r = stokes_einstein_radius(D=1e-12, eta=WATER_VISCOSITY_25C, T=298.15)
        self.assertIsInstance(r, float)
        self.assertGreater(r, 0)

    def test_stokes_einstein_viscosity_scalar(self):
        """Test Stokes-Einstein viscosity with scalar input."""
        eta = stokes_einstein_viscosity(D=1e-9, r=1e-9, T=298.15)
        self.assertIsInstance(eta, float)
        self.assertGreater(eta, 0)

    def test_stokes_einstein_rotational_diffusion_scalar(self):
        """Test Stokes-Einstein rotational diffusion with scalar input."""
        Dr = stokes_einstein_rotational_diffusion(r=1e-9, eta=WATER_VISCOSITY_25C, T=298.15)
        self.assertIsInstance(Dr, float)
        self.assertGreater(Dr, 0)

    def test_water_viscosity_constant(self):
        """Test that water viscosity constant exists."""
        self.assertIsInstance(WATER_VISCOSITY_25C, float)
        self.assertAlmostEqual(WATER_VISCOSITY_25C, 8.9e-4, places=6)

    def test_inverse_relationship(self):
        """Test inverse relationship between diffusion and radius."""
        r_orig = 1e-9
        D = stokes_einstein_diffusion(r=r_orig, eta=WATER_VISCOSITY_25C, T=298.15)
        r_back = stokes_einstein_radius(D=D, eta=WATER_VISCOSITY_25C, T=298.15)
        self.assertAlmostEqual(r_orig, r_back, places=15)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(RadiusM)
        self.assertIsNotNone(ViscosityPaS)
        self.assertIsNotNone(DiffusionCoefficientM2S)

    def test_normalize_radius_various_units(self):
        """Test normalize_radius with various unit inputs."""
        test_cases = [
            ("1 m", 1.0),
            ("1 nm", 1e-9),
            ("1 µm", 1e-6),
            ("1 Å", 1e-10),
            ("1 cm", 1e-2),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_radius(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_viscosity_various_units(self):
        """Test normalize_viscosity with various unit inputs."""
        test_cases = [
            ("1 Pa·s", 1.0),
            ("1 Pa*s", 1.0),
            ("1 cP", 1e-3),
            ("1 P", 0.1),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_viscosity(input_val)
                self.assertAlmostEqual(result, expected)

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

    def test_stokes_einstein_functions_various_units(self):
        """Test Stokes-Einstein functions with various unit inputs."""
        # Test with different radius units
        D1 = stokes_einstein_diffusion("1 nm", eta=WATER_VISCOSITY_25C, T=298.15)
        D2 = stokes_einstein_diffusion("1000 pm", eta=WATER_VISCOSITY_25C, T=298.15)
        self.assertAlmostEqual(D1, D2)


if __name__ == "__main__":
    unittest.main()

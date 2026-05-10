#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.Diffusion import (
    fick_first_law,
    fick_diffusion_distance,
    fick_diffusion_time,
    fick_semi_infinite_concentration,
    fick_thin_film_concentration,
    diffusion_coefficient_from_temperature,
    normalize_diffusion_coefficient, DiffusionCoefficientM2S,
    normalize_time_seconds, TimeSeconds,
    normalize_length, LengthMeter,
    normalize_concentration, ConcentrationMolM3,
    normalize_energy, EnergyJPerMol,
)
import numpy as np


class TestFickLaws(unittest.TestCase):
    def test_fick_first_law_scalar(self):
        """Test Fick's first law with scalar input"""
        flux = fick_first_law(D=1e-9, dC_dx=-100.0)
        self.assertIsInstance(flux, float)

    def test_fick_diffusion_distance_scalar(self):
        """Test diffusion distance with scalar input"""
        distance = fick_diffusion_distance(D=1e-9, t=3600.0)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)

    def test_fick_diffusion_time_scalar(self):
        """Test diffusion time with scalar input"""
        time = fick_diffusion_time(D=1e-9, x=0.001)
        self.assertIsInstance(time, float)
        self.assertGreater(time, 0)

    def test_fick_semi_infinite_concentration_scalar(self):
        """Test semi-infinite concentration with scalar input"""
        conc = fick_semi_infinite_concentration(C0=0.0, Cs=1.0, x=0.001, D=1e-9, t=3600.0)
        self.assertIsInstance(conc, float)

    def test_fick_thin_film_concentration_scalar(self):
        """Test thin film concentration with scalar input"""
        conc = fick_thin_film_concentration(M=1.0, D=1e-9, t=3600.0, x=0.001)
        self.assertIsInstance(conc, float)

    def test_diffusion_coefficient_from_temperature_scalar(self):
        """Test temperature dependence of diffusion coefficient"""
        D = diffusion_coefficient_from_temperature(D0=1e-9, Ea=50000.0, T=298.15)
        self.assertIsInstance(D, float)
        self.assertGreater(D, 0)

    def test_inverse_diffusion_distance_time(self):
        """Test inverse relationship between distance and time"""
        D = 1e-9
        t = 3600.0
        x = fick_diffusion_distance(D, t)
        t_calc = fick_diffusion_time(D, x)
        self.assertAlmostEqual(t, t_calc, places=10)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(DiffusionCoefficientM2S)
        self.assertIsNotNone(TimeSeconds)
        self.assertIsNotNone(LengthMeter)
        self.assertIsNotNone(ConcentrationMolM3)
        self.assertIsNotNone(EnergyJPerMol)

    def test_normalize_diffusion_coefficient_various_units(self):
        """Test normalize_diffusion_coefficient with various unit inputs"""
        test_cases = [
            ("1 m²/s", 1.0),
            ("1 m2/s", 1.0),
            ("1 cm²/s", 1e-4),
            ("1 cm2/s", 1e-4),
            ("1 mm²/s", 1e-6),
            ("1 mm2/s", 1e-6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_diffusion_coefficient(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_time_seconds_various_units(self):
        """Test normalize_time_seconds with various unit inputs"""
        test_cases = [
            ("1 s", 1.0),
            ("1 ms", 1e-3),
            ("1 µs", 1e-6),
            ("1 ns", 1e-9),
            ("1 min", 60.0),
            ("1 h", 3600.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_time_seconds(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_length_various_units(self):
        """Test normalize_length with various unit inputs"""
        test_cases = [
            ("1 m", 1.0),
            ("1 mm", 1e-3),
            ("1 cm", 1e-2),
            ("1 km", 1e3),
            ("1 µm", 1e-6),
            ("1 nm", 1e-9),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_length(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_concentration_various_units(self):
        """Test normalize_concentration with various unit inputs"""
        test_cases = [
            ("1 mol/m³", 1.0),
            ("1 mol/m3", 1.0),
            ("1 mol/m^3", 1.0),
            ("1 mol/L", 1000.0),
            ("1 M", 1000.0),
            ("1 mM", 1.0),
            ("1 µM", 1e-3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_concentration(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_energy_various_units(self):
        """Test normalize_energy with various unit inputs"""
        test_cases = [
            ("1 J/mol", 1.0),
            ("1 kJ/mol", 1000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_energy(input_val)
                self.assertAlmostEqual(result, expected)

    def test_fick_functions_various_units(self):
        """Test Fick's laws functions with various unit inputs"""
        # Test fick_diffusion_distance with different time units
        x1 = fick_diffusion_distance("1e-9 m²/s", "3600 s")
        x2 = fick_diffusion_distance("1e-9 m²/s", "1 h")
        self.assertAlmostEqual(x1, x2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.IonExchange import (
    selectivity_coefficient,
    separation_factor,
    distribution_coefficient,
    ion_exchange_capacity_from_breakthrough,
    donnan_potential,
    donnan_ratio,
    normalize_concentration, ConcentrationMolar,
    normalize_volume, VolumeLiter,
    normalize_mass, MassGram,
)
import numpy as np


class TestIonExchange(unittest.TestCase):
    def test_selectivity_coefficient_scalar(self):
        """Test selectivity coefficient with scalar input"""
        K = selectivity_coefficient(q_A=0.1, C_B=0.1, q_B=0.05, C_A=0.1)
        self.assertIsInstance(K, float)
        self.assertGreater(K, 0)

    def test_separation_factor_scalar(self):
        """Test separation factor with scalar input"""
        alpha = separation_factor(q_A=0.1, C_B=0.1, q_B=0.05, C_A=0.1)
        self.assertIsInstance(alpha, float)
        self.assertGreater(alpha, 1.0)

    def test_distribution_coefficient_scalar(self):
        """Test distribution coefficient with scalar input"""
        Kd = distribution_coefficient(q=0.1, C=0.01)
        self.assertIsInstance(Kd, float)
        self.assertGreater(Kd, 0)

    def test_ion_exchange_capacity_from_breakthrough_scalar(self):
        """Test ion exchange capacity from breakthrough with scalar input"""
        Q = ion_exchange_capacity_from_breakthrough(C_feed=0.1, V_breakthrough=10.0, mass_resin=100.0)
        self.assertIsInstance(Q, float)
        self.assertGreater(Q, 0)

    def test_donnan_potential_scalar(self):
        """Test Donnan potential with scalar input"""
        E = donnan_potential(z=1, C_in=0.1, C_out=1.0)
        self.assertIsInstance(E, float)

    def test_donnan_ratio_scalar(self):
        """Test Donnan ratio with scalar input"""
        r = donnan_ratio(z=1, C_fixed=0.1, C_solution=0.1)
        self.assertIsInstance(r, float)
        self.assertGreater(r, 0)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(ConcentrationMolar)
        self.assertIsNotNone(VolumeLiter)
        self.assertIsNotNone(MassGram)

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

    def test_normalize_volume_various_units(self):
        """Test normalize_volume with various unit inputs"""
        test_cases = [
            ("1 L", 1.0),
            ("1 mL", 1e-3),
            ("1 µL", 1e-6),
            ("1 m³", 1000.0),
            ("1 cm³", 1e-3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_volume(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_mass_various_units(self):
        """Test normalize_mass with various unit inputs"""
        test_cases = [
            ("1 g", 1.0),
            ("1 mg", 1e-3),
            ("1 µg", 1e-6),
            ("1 kg", 1000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_mass(input_val)
                self.assertAlmostEqual(result, expected)

    def test_ion_exchange_functions_various_units(self):
        """Test ion exchange functions with various unit inputs"""
        # Test with different concentration units
        K1 = selectivity_coefficient("0.1 M", "0.1 M", "0.05 M", "0.1 M")
        K2 = selectivity_coefficient("100 mM", "100 mM", "50 mM", "100 mM")
        self.assertAlmostEqual(K1, K2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.Stoichiometry import (
    parse_formula,
    molecular_weight,
    moles_to_grams,
    grams_to_moles,
    moles_to_particles,
    particles_to_moles,
    molarity_from_moles_volume,
    moles_from_molarity_volume,
    volume_from_molarity_moles,
    molality_from_moles_mass,
    dilution_volume,
    mass_fraction_to_molarity,
    limiting_reagent,
    theoretical_yield,
    percent_yield,
    empirical_formula_from_percent,
    ideal_gas_moles,
    ideal_gas_volume,
    ATOMIC_WEIGHTS,
    normalize_moles, Moles,
    normalize_grams, Grams,
    normalize_volume, VolumeLiter,
    normalize_pressure, PressurePa,
    normalize_density, DensityKgL,
    normalize_molar_mass, MolarMassGMol,
)


class TestStoichiometry(unittest.TestCase):
    def test_parse_formula_simple(self):
        """Test parsing simple formulas"""
        result = parse_formula("H2O")
        self.assertEqual(result, {"H": 2, "O": 1})

    def test_parse_formula_parentheses(self):
        """Test parsing formulas with parentheses"""
        result = parse_formula("Ca(OH)2")
        self.assertEqual(result, {"Ca": 1, "O": 2, "H": 2})

    def test_molecular_weight(self):
        """Test molecular weight calculation"""
        mw = molecular_weight("H2O")
        self.assertAlmostEqual(mw, 18.015, places=2)

    def test_moles_to_grams_scalar(self):
        """Test moles to grams conversion"""
        result = moles_to_grams(moles=1.0, molar_mass=18.015)
        self.assertAlmostEqual(result, 18.015, places=2)

    def test_grams_to_moles_scalar(self):
        """Test grams to moles conversion."""
        result = grams_to_moles(grams=18.015, molar_mass=18.015)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_moles_to_particles_scalar(self):
        """Test moles to particles conversion"""
        result = moles_to_particles(moles=1.0)
        self.assertIsInstance(result, float)

    def test_molarity_from_moles_volume_scalar(self):
        """Test molarity calculation."""
        result = molarity_from_moles_volume(moles=1.0, volume_liters=1.0)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_ideal_gas_moles_scalar(self):
        """Test ideal gas moles calculation"""
        result = ideal_gas_moles(pressure_Pa=101325.0, volume_m3=0.022414, T=273.15)
        self.assertAlmostEqual(result, 1.0, places=3)

    def test_ideal_gas_volume_scalar(self):
        """Test ideal gas volume calculation"""
        result = ideal_gas_volume(moles=1.0, T=273.15, pressure_Pa=101325.0)
        self.assertAlmostEqual(result, 0.022414, places=5)

    def test_atomic_weights_dict(self):
        """Test that atomic weights dictionary exists"""
        self.assertIsInstance(ATOMIC_WEIGHTS, dict)
        self.assertIn("H", ATOMIC_WEIGHTS)
        self.assertIn("C", ATOMIC_WEIGHTS)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(Moles)
        self.assertIsNotNone(Grams)
        self.assertIsNotNone(VolumeLiter)
        self.assertIsNotNone(PressurePa)
        self.assertIsNotNone(DensityKgL)
        self.assertIsNotNone(MolarMassGMol)

    def test_normalize_moles_various_units(self):
        """Test normalize_moles with various unit inputs"""
        test_cases = [
            ("1 mol", 1.0),
            ("1 mmol", 1e-3),
            ("1 µmol", 1e-6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_moles(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_grams_various_units(self):
        """Test normalize_grams with various unit inputs."""
        test_cases = [
            ("1 g", 1.0),
            ("1 mg", 1e-3),
            ("1 kg", 1e3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_grams(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_volume_various_units(self):
        """Test normalize_volume with various unit inputs."""
        test_cases = [
            ("1 L", 1.0),
            ("1 mL", 1e-3),
            ("1 m³", 1000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_volume(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_pressure_various_units(self):
        """Test normalize_pressure with various unit inputs"""
        test_cases = [
            ("1 Pa", 1.0),
            ("1 kPa", 1e3),
            ("1 atm", 101325.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_pressure(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_density_various_units(self):
        """Test normalize_density with various unit inputs"""
        test_cases = [
            ("1 kg/L", 1.0),
            ("1 g/mL", 1.0),
            ("1 g/cm³", 1.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_density(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_molar_mass_various_units(self):
        """Test normalize_molar_mass with various unit inputs"""
        test_cases = [
            ("1 g/mol", 1.0),
            ("1 kg/mol", 1000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_molar_mass(input_val)
                self.assertAlmostEqual(result, expected)

    def test_stoichiometry_functions_various_units(self):
        """Test stoichiometry functions with various unit inputs"""
        # Test moles_to_grams with different units
        m1 = moles_to_grams("1 mol", "18.015 g/mol")
        m2 = moles_to_grams("1000 mmol", "18.015 g/mol")
        self.assertAlmostEqual(m1, m2)


if __name__ == "__main__":
    unittest.main()

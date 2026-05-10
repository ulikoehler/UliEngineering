#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.Henderson import (
    henderson_hasselbalch_pH,
    henderson_hasselbalch_ratio,
    henderson_hasselbalch_pKa,
    henderson_junction_potential,
    henderson_junction_potential_simple,
    buffer_capacity,
    normalize_concentration, ConcentrationMolar,
    normalize_molar_conductivity, MolarConductivitySM2Mol,
)


class TestHendersonHasselbalch(unittest.TestCase):
    def test_henderson_hasselbalch_pH_scalar(self):
        """Test Henderson-Hasselbalch pH calculation with scalar input"""
        pH = henderson_hasselbalch_pH(pKa=4.75, base_concentration=0.1, acid_concentration=0.1)
        self.assertIsInstance(pH, float)
        self.assertAlmostEqual(pH, 4.75, places=6)

    def test_henderson_hasselbalch_ratio_scalar(self):
        """Test Henderson-Hasselbalch ratio calculation with scalar input"""
        ratio = henderson_hasselbalch_ratio(pH=4.75, pKa=4.75)
        self.assertIsInstance(ratio, float)
        self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_henderson_hasselbalch_pKa_scalar(self):
        """Test Henderson-Hasselbalch pKa calculation with scalar input"""
        pKa = henderson_hasselbalch_pKa(pH=4.75, base_concentration=0.1, acid_concentration=0.1)
        self.assertIsInstance(pKa, float)
        self.assertAlmostEqual(pKa, 4.75, places=6)

    def test_henderson_junction_potential_scalar(self):
        """Test Henderson junction potential with scalar input"""
        E = henderson_junction_potential(t_plus=0.49, t_minus=0.51, c1=0.1, c2=1.0)
        self.assertIsInstance(E, float)

    def test_henderson_junction_potential_simple_scalar(self):
        """Test Henderson junction potential simple with scalar input"""
        E = henderson_junction_potential_simple(lambda_plus=0.01, lambda_minus=0.01, c1=0.1, c2=1.0)
        self.assertIsInstance(E, float)

    def test_buffer_capacity_scalar(self):
        """Test buffer capacity calculation with scalar input"""
        beta = buffer_capacity(C_total=0.1, Ka=1e-4, H_concentration=1e-4)
        self.assertIsInstance(beta, float)
        self.assertGreater(beta, 0)

    def test_inverse_hh_equation(self):
        """Test inverse relationship between pH and ratio functions"""
        pKa = 4.75
        base_conc = 0.1
        acid_conc = 0.1
        pH = henderson_hasselbalch_pH(pKa, base_conc, acid_conc)
        ratio = henderson_hasselbalch_ratio(pH, pKa)
        ratio_calc = base_conc / acid_conc
        self.assertAlmostEqual(ratio, ratio_calc, places=10)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(ConcentrationMolar)
        self.assertIsNotNone(MolarConductivitySM2Mol)

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

    def test_normalize_molar_conductivity_various_units(self):
        """Test normalize_molar_conductivity with various unit inputs"""
        test_cases = [
            ("1 S·m²/mol", 1.0),
            ("1 S·cm²/mol", 1e-4),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_molar_conductivity(input_val)
                self.assertAlmostEqual(result, expected)

    def test_henderson_functions_various_units(self):
        """Test Henderson functions with various unit inputs"""
        # Test with different concentration units
        pH1 = henderson_hasselbalch_pH(4.75, "0.1 M", "0.1 M")
        pH2 = henderson_hasselbalch_pH(4.75, "100 mM", "100 mM")
        self.assertAlmostEqual(pH1, pH2)


if __name__ == "__main__":
    unittest.main()

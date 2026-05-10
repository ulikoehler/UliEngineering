#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.ElectrolyteConductivity import (
    electrolyte_conductivity_from_molar,
    electrolyte_resistivity,
    conductivity_from_cell_constant,
    molar_conductivity_from_conductivity,
    specific_conductance_temperature_correction,
    normalize_molar_conductivity, MolarConductivitySCm2Mol,
    normalize_concentration, ConcentrationMolar,
    normalize_conductivity, ConductivitySM,
    normalize_conductance, ConductanceS,
    normalize_cell_constant, CellConstantPerMeter,
)
import numpy as np


class TestElectrolyteConductivity(unittest.TestCase):
    def test_electrolyte_conductivity_from_molar_scalar(self):
        """Test conductivity from molar conductivity with scalar input."""
        kappa = electrolyte_conductivity_from_molar(Lambda_m=100.0, c=0.1)
        self.assertIsInstance(kappa, float)
        self.assertAlmostEqual(kappa, 1.0, places=6)

    def test_electrolyte_resistivity_scalar(self):
        """Test resistivity from conductivity with scalar input."""
        rho = electrolyte_resistivity(conductivity=10.0)
        self.assertIsInstance(rho, float)
        self.assertAlmostEqual(rho, 0.1, places=6)

    def test_conductivity_from_cell_constant_scalar(self):
        """Test conductivity from cell constant with scalar input."""
        kappa = conductivity_from_cell_constant(conductance=0.01, cell_constant=100.0)
        self.assertIsInstance(kappa, float)
        self.assertAlmostEqual(kappa, 1.0, places=6)

    def test_molar_conductivity_from_conductivity_scalar(self):
        """Test molar conductivity from conductivity with scalar input."""
        Lambda_m = molar_conductivity_from_conductivity(conductivity=10.0, c=0.1)
        self.assertIsInstance(Lambda_m, float)
        self.assertAlmostEqual(Lambda_m, 1000.0, places=6)

    def test_specific_conductance_temperature_correction_scalar(self):
        """Test temperature correction with scalar input."""
        kappa = specific_conductance_temperature_correction(kappa_ref=1.0, T=308.15, T_ref=298.15, alpha=0.02)
        self.assertIsInstance(kappa, float)
        self.assertAlmostEqual(kappa, 1.2, places=6)

    def test_inverse_conductivity_resistivity(self):
        """Test inverse relationship between conductivity and resistivity."""
        kappa = 10.0
        rho = electrolyte_resistivity(kappa)
        kappa_calc = 1.0 / rho
        self.assertAlmostEqual(kappa, kappa_calc, places=10)

    def test_inverse_molar_conductivity(self):
        """Test inverse relationship between molar conductivity functions."""
        Lambda_m = 100.0
        c = 0.1
        kappa = electrolyte_conductivity_from_molar(Lambda_m, c)
        Lambda_m_calc = molar_conductivity_from_conductivity(kappa, c)
        self.assertAlmostEqual(Lambda_m, Lambda_m_calc, places=10)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(MolarConductivitySCm2Mol)
        self.assertIsNotNone(ConcentrationMolar)
        self.assertIsNotNone(ConductivitySM)
        self.assertIsNotNone(ConductanceS)
        self.assertIsNotNone(CellConstantPerMeter)

    def test_normalize_molar_conductivity_various_units(self):
        """Test normalize_molar_conductivity with various unit inputs."""
        test_cases = [
            ("1 S·cm²/mol", 1.0),
            ("1 S·m²/mol", 10000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_molar_conductivity(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_concentration_various_units(self):
        """Test normalize_concentration with various unit inputs."""
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

    def test_normalize_conductivity_various_units(self):
        """Test normalize_conductivity with various unit inputs."""
        test_cases = [
            ("1 S/m", 1.0),
            ("1 S/cm", 100.0),
            ("1 mS/m", 1e-3),
            ("1 µS/m", 1e-6),
            ("1 µS/cm", 0.1),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_conductivity(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_conductance_various_units(self):
        """Test normalize_conductance with various unit inputs."""
        test_cases = [
            ("1 S", 1.0),
            ("1 mS", 1e-3),
            ("1 µS", 1e-6),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_conductance(input_val)
                self.assertAlmostEqual(result, expected)

    def test_normalize_cell_constant_various_units(self):
        """Test normalize_cell_constant with various unit inputs."""
        test_cases = [
            ("1 1/m", 1.0),
            ("1 m⁻¹", 1.0),
            ("1 1/cm", 100.0),
            ("1 cm⁻¹", 100.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_cell_constant(input_val)
                self.assertAlmostEqual(result, expected)

    def test_electrolyte_functions_various_units(self):
        """Test electrolyte conductivity functions with various unit inputs."""
        # Test with different concentration units
        k1 = electrolyte_conductivity_from_molar("100 S·cm²/mol", "0.1 M")
        k2 = electrolyte_conductivity_from_molar("100 S·cm²/mol", "100 mM")
        self.assertAlmostEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()

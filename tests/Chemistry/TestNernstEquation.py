#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numpy.testing import assert_approx_equal
from UliEngineering.Chemistry.NernstEquation import (
    nernst_cell_potential,
    nernst_half_cell_potential,
    nernst_potential_at_25C,
    nernst_reaction_quotient_from_potential,
    FARADAY_CONSTANT,
    normalize_concentration, ConcentrationMolar,
)
from scipy.constants import R as gas_constant


class TestNernstCellPotential(unittest.TestCase):
    def test_standard_conditions(self):
        # Q=1 => E = E0
        assert_approx_equal(nernst_cell_potential(1.10, 2, 1.0), 1.10, significant=5)

    def test_q_greater_than_1(self):
        # Q > 1 => E < E0
        E = nernst_cell_potential(1.10, 2, 10.0)
        self.assertLess(E, 1.10)

    def test_q_less_than_1(self):
        # Q < 1 => E > E0
        E = nernst_cell_potential(1.10, 2, 0.1)
        self.assertGreater(E, 1.10)

    def test_known_value(self):
        # E = 1.10 - (8.314*298.15)/(2*96485.33) * ln(10) = 1.10 - 0.02957
        E = nernst_cell_potential(1.10, 2, 10.0, 298.15)
        thermal_voltage = gas_constant * 298.15 / (2 * FARADAY_CONSTANT)
        expected = 1.10 - thermal_voltage * np.log(10.0)
        assert_approx_equal(E, expected, significant=5)

    def test_different_temperature(self):
        E_cold = nernst_cell_potential(1.10, 2, 10.0, 273.15)
        E_hot = nernst_cell_potential(1.10, 2, 10.0, 373.15)
        # Higher temp => larger deviation from E0
        self.assertGreater(abs(1.10 - E_hot), abs(1.10 - E_cold))

    def test_single_electron_transfer(self):
        E = nernst_cell_potential(0.80, 1, 0.01, 298.15)
        expected = 0.80 - (gas_constant * 298.15 / FARADAY_CONSTANT) * np.log(0.01)
        assert_approx_equal(E, expected, significant=5)


class TestNernstHalfCellPotential(unittest.TestCase):
    def test_equal_concentrations(self):
        # Equal concentrations => E = E0 (Q=1)
        E = nernst_half_cell_potential(0.34, 2, 0.1, 0.1)
        assert_approx_equal(E, 0.34, significant=5)

    def test_asymmetric(self):
        E = nernst_half_cell_potential(0.34, 2, 0.1, 1.0)
        self.assertLess(E, 0.34)

    def test_inverse(self):
        E = nernst_half_cell_potential(0.34, 2, 1.0, 0.1)
        self.assertGreater(E, 0.34)


class TestNernstPotentialAt25C(unittest.TestCase):
    def test_standard(self):
        assert_approx_equal(nernst_potential_at_25C(1.10, 2, 1.0), 1.10, significant=5)

    def test_known_value(self):
        E = nernst_potential_at_25C(1.10, 2, 10.0)
        expected = 1.10 - (0.025693 / 2) * np.log(10.0)
        assert_approx_equal(E, expected, significant=4)


class TestNernstReactionQuotient(unittest.TestCase):
    def test_at_standard(self):
        Q = nernst_reaction_quotient_from_potential(1.10, 1.10, 2, 298.15)
        assert_approx_equal(Q, 1.0, significant=4)

    def test_roundtrip(self):
        E0, n, Q_orig = 0.76, 2, 5.0
        E = nernst_cell_potential(E0, n, Q_orig, 298.15)
        Q_back = nernst_reaction_quotient_from_potential(E, E0, n, 298.15)
        assert_approx_equal(Q_back, Q_orig, significant=4)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(ConcentrationMolar)

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

    def test_nernst_functions_various_units(self):
        """Test Nernst functions with various unit inputs"""
        # Test with different concentration units
        E1 = nernst_half_cell_potential(0.34, 2, "0.1 M", "0.1 M")
        E2 = nernst_half_cell_potential(0.34, 2, "100 mM", "100 mM")
        self.assertAlmostEqual(E1, E2)


if __name__ == '__main__':
    unittest.main()

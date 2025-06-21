#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.EngineerIO.Concentration import *

class TestMassConcentration(unittest.TestCase):
    def setUp(self):
        self.mass_io = EngineerMassConcentrationIO()

    def test_basic_normalization(self):
        assert_approx_equal(normalize_mass_concentration(1.0), 1.0)
        assert_approx_equal(self.mass_io.normalize_mass_concentration(5.0), 5.0)
        assert_approx_equal(normalize_mass_concentration(3), 3)
        self.assertIsNone(normalize_mass_concentration(None))
        self.assertIsNone(self.mass_io.normalize_mass_concentration(None))

    def test_mass_units(self):
        assert_approx_equal(normalize_mass_concentration("1.0 g/l"), 1.0)
        assert_approx_equal(normalize_mass_concentration("1.0 mg/l"), 1e-3)
        assert_approx_equal(normalize_mass_concentration("1.0 µg/l"), 1e-6)
        assert_approx_equal(normalize_mass_concentration("1.0 ng/l"), 1e-9)
        assert_approx_equal(normalize_mass_concentration("1.0 g/ml"), 1000.0)
        assert_approx_equal(normalize_mass_concentration("1.0 mg/ml"), 1.0)
        assert_approx_equal(normalize_mass_concentration("1.0 µg/ml"), 1e-3)
        assert_approx_equal(normalize_mass_concentration("1.0 ng/ml"), 1e-6)
        assert_approx_equal(normalize_mass_concentration("1.0 pg/ml"), 1e-9)
        assert_approx_equal(normalize_mass_concentration("1.0 g/µl"), 1e6)
        assert_approx_equal(normalize_mass_concentration("1.0 mg/µl"), 1e3)
        assert_approx_equal(normalize_mass_concentration("1.0 µg/µl"), 1.0)
        assert_approx_equal(normalize_mass_concentration("1.0 ng/µl"), 1e-3)
        assert_approx_equal(normalize_mass_concentration("1.0 pg/µl"), 1e-6)
        assert_approx_equal(normalize_mass_concentration("1.0 g/nl"), 1e9)
        assert_approx_equal(normalize_mass_concentration("1.0 mg/nl"), 1e6)
        assert_approx_equal(normalize_mass_concentration("1.0 µg/nl"), 1e3)
        assert_approx_equal(normalize_mass_concentration("1.0 ng/nl"), 1.0)
        assert_approx_equal(normalize_mass_concentration("1.0 pg/nl"), 1e-3)

    def test_mass_concentration_percent_ppm(self):
        assert_approx_equal(normalize_mass_concentration("1 %"), 1e-2)
        assert_approx_equal(normalize_mass_concentration("1 ppm"), 1e-6)
        assert_approx_equal(normalize_mass_concentration("1 ppb"), 1e-9)

    def test_mass_concentration_lists(self):
        input_list = ["1 g/l", "1000 mg/l", "1 mg/ml"]
        expected = [1.0, 1.0, 1.0]
        result = normalize_mass_concentration(input_list)
        for r, e in zip(result, expected):
            assert_approx_equal(r, e)

    def test_mass_concentration_numpy_arrays(self):
        arr = np.array(["1 g/l", "1000 mg/l", "1 mg/ml"])
        expected = np.array([1.0, 1.0, 1.0])
        result = normalize_mass_concentration(arr)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_mass_concentration(self):
        assert_approx_equal(convert_mass_concentration_to_per_liter(1.0, "g/l"), 1.0)
        assert_approx_equal(convert_mass_concentration_to_per_liter(1000, "mg/l"), 1.0)
        assert_approx_equal(convert_mass_concentration_to_per_liter(1, "mg/ml"), 1.0)
        assert_approx_equal(convert_mass_concentration_to_per_liter(1, "%"), 1e-2)

    def test_consistency_between_functions(self):
        test_cases = [
            (1.0, "g/l"), (1000.0, "mg/l"), (1.0, "mg/ml"), (1.0, "%")
        ]
        for value, unit in test_cases:
            result1 = normalize_mass_concentration(f"{value} {unit}")
            result2 = convert_mass_concentration_to_per_liter(value, unit)
            assert_approx_equal(result1, result2)

    @parameterized.expand([
        ("1A",), ("xyz",), ("invalid_unit",), ("square invalid",),
    ])
    def test_invalid_units(self, unit):
        with self.assertRaises(Exception):
            normalize_mass_concentration(f"6.6 {unit}")

    def test_empty_string(self):
        with self.assertRaises(Exception):
            normalize_mass_concentration("")
        with self.assertRaises(Exception):
            normalize_mass_concentration("   ")

class TestAmountConcentration(unittest.TestCase):
    def setUp(self):
        self.amount_io = EngineerAmountConcentrationIO()

    def test_basic_normalization(self):
        assert_approx_equal(normalize_amount_concentration(1.0), 1.0)
        assert_approx_equal(self.amount_io.normalize_amount_concentration(5.0), 5.0)
        assert_approx_equal(normalize_amount_concentration(3), 3)
        self.assertIsNone(normalize_amount_concentration(None))
        self.assertIsNone(self.amount_io.normalize_amount_concentration(None))

    def test_amount_units(self):
        assert_approx_equal(normalize_amount_concentration("1.0 mol/l"), 1.0)
        assert_approx_equal(normalize_amount_concentration("1.0 mmol/l"), 1e-3)
        assert_approx_equal(normalize_amount_concentration("1.0 µmol/l"), 1e-6)
        assert_approx_equal(normalize_amount_concentration("1.0 nmol/l"), 1e-9)
        assert_approx_equal(normalize_amount_concentration("1.0 pmol/l"), 1e-12)
        assert_approx_equal(normalize_amount_concentration("1.0 mol/ml"), 1000.0)
        assert_approx_equal(normalize_amount_concentration("1.0 mmol/ml"), 1.0)
        assert_approx_equal(normalize_amount_concentration("1.0 µmol/ml"), 1e-3)
        assert_approx_equal(normalize_amount_concentration("1.0 nmol/ml"), 1e-6)
        assert_approx_equal(normalize_amount_concentration("1.0 pmol/ml"), 1e-9)
        assert_approx_equal(normalize_amount_concentration("1.0 mol/µl"), 1e6)
        assert_approx_equal(normalize_amount_concentration("1.0 mmol/µl"), 1e3)
        assert_approx_equal(normalize_amount_concentration("1.0 µmol/µl"), 1.0)
        assert_approx_equal(normalize_amount_concentration("1.0 nmol/µl"), 1e-3)
        assert_approx_equal(normalize_amount_concentration("1.0 pmol/µl"), 1e-6)
        assert_approx_equal(normalize_amount_concentration("1.0 mol/nl"), 1e9)
        assert_approx_equal(normalize_amount_concentration("1.0 mmol/nl"), 1e6)
        assert_approx_equal(normalize_amount_concentration("1.0 µmol/nl"), 1e3)
        assert_approx_equal(normalize_amount_concentration("1.0 nmol/nl"), 1.0)
        assert_approx_equal(normalize_amount_concentration("1.0 pmol/nl"), 1e-3)

    def test_amount_concentration_percent_ppm(self):
        assert_approx_equal(normalize_amount_concentration("1 %"), 0.01)
        assert_approx_equal(normalize_amount_concentration("1 ppm"), 1e-6)
        assert_approx_equal(normalize_amount_concentration("1 ppb"), 1e-9)

    def test_amount_concentration_lists(self):
        input_list = ["1 mol/l", "1000 mmol/l", "1 mmol/ml"]
        expected = [1.0, 1.0, 1.0]
        result = normalize_amount_concentration(input_list)
        for r, e in zip(result, expected):
            assert_approx_equal(r, e)

    def test_amount_concentration_numpy_arrays(self):
        arr = np.array(["1 mol/l", "1000 mmol/l", "1 mmol/ml"])
        expected = np.array([1.0, 1.0, 1.0])
        result = normalize_amount_concentration(arr)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_amount_concentration(self):
        assert_approx_equal(convert_amount_concentration_to_grams_per_liter(1.0, "mol/l"), 1.0)
        assert_approx_equal(convert_amount_concentration_to_grams_per_liter(1000, "mmol/l"), 1.0)
        assert_approx_equal(convert_amount_concentration_to_grams_per_liter(1, "mmol/ml"), 1.0)
        assert_approx_equal(convert_amount_concentration_to_grams_per_liter(1, "%"), 0.01)

    def test_consistency_between_functions(self):
        test_cases = [
            (1.0, "mol/l"), (1000.0, "mmol/l"), (1.0, "mmol/ml"), (1.0, "%")
        ]
        for value, unit in test_cases:
            result1 = normalize_amount_concentration(f"{value} {unit}")
            result2 = convert_amount_concentration_to_grams_per_liter(value, unit)
            assert_approx_equal(result1, result2)

    @parameterized.expand([
        ("1A",), ("xyz",), ("invalid_unit",), ("square invalid",),
    ])
    def test_invalid_units(self, unit):
        with self.assertRaises(Exception):
            normalize_amount_concentration(f"6.6 {unit}")

    def test_empty_string(self):
        with self.assertRaises(Exception):
            normalize_amount_concentration("")
        with self.assertRaises(Exception):
            normalize_amount_concentration("   ")

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.DNA import (
    dna_molecular_weight,
    NucleotideFractions,
    NucleotideWeights,
    human_dna_fractions,
    equal_dna_fractions,
    dna_weight_concentration_from_concentration
)
import numpy as np

class TestDNAMolecularWeight(unittest.TestCase):
    def test_human_dna_fractions(self):
        # 100 nt, human fractions, default weights
        mw = dna_molecular_weight(100, human_dna_fractions)
        expected = (
            100 * 0.3 * 313.2 +
            100 * 0.3 * 304.2 +
            100 * 0.2 * 329.2 +
            100 * 0.2 * 289.2 +
            79.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_equal_fractions(self):
        # 40 nt, equal fractions, default weights
        mw = dna_molecular_weight(40, equal_dna_fractions)
        expected = (
            10 * 313.2 +
            10 * 304.2 +
            10 * 329.2 +
            10 * 289.2 +
            79.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_custom_fractions(self):
        # Custom fractions, 50 nt
        fractions = NucleotideFractions(A=0.5, T=0.2, G=0.2, C=0.1)
        mw = dna_molecular_weight(50, fractions)
        expected = (
            25 * 313.2 +
            10 * 304.2 +
            10 * 329.2 +
            5 * 289.2 +
            79.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_custom_weights(self):
        # Use custom weights
        custom_weights = NucleotideWeights(A=314, T=305, G=330, C=290)
        fractions = NucleotideFractions(A=0.25, T=0.25, G=0.25, C=0.25)
        mw = dna_molecular_weight(4, fractions, custom_weights)
        expected = (
            1 * 314 +
            1 * 305 +
            1 * 330 +
            1 * 290 +
            79.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_zero_length(self):
        # Zero length DNA
        mw = dna_molecular_weight(0, equal_dna_fractions)
        self.assertAlmostEqual(mw, 79.0, places=6)

    def test_sum_of_fractions(self):
        # Fractions not summing to 1.0
        fractions = NucleotideFractions(A=0.5, T=0.3, G=0.1, C=0.1)
        mw = dna_molecular_weight(10, fractions)
        expected = (
            5 * 313.2 +
            3 * 304.2 +
            1 * 329.2 +
            1 * 289.2 +
            79.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

class TestDNAWeightConcentration(unittest.TestCase):
    def test_mol_per_liter(self):
        mw = dna_molecular_weight(1, equal_dna_fractions)
        result = dna_weight_concentration_from_concentration("1 mol/l", 1, equal_dna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertAlmostEqual(result[0], mw, places=6)
        else:
            self.assertAlmostEqual(result, mw, places=6)

    def test_micromolar(self):
        mw = dna_molecular_weight(10, equal_dna_fractions)
        result = dna_weight_concentration_from_concentration("1 uM", 10, equal_dna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertAlmostEqual(result[0], 1e-6 * mw, places=12)
        else:
            self.assertAlmostEqual(result, 1e-6 * mw, places=12)

    def test_list_input(self):
        mw = dna_molecular_weight(5, equal_dna_fractions)
        result = dna_weight_concentration_from_concentration(["1 uM", "2 uM"], 5, equal_dna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertEqual(len(result), 2)
            self.assertAlmostEqual(result[0], 1e-6 * mw, places=12)
            self.assertAlmostEqual(result[1], 2e-6 * mw, places=12)
        else:
            self.fail("Expected list or ndarray output")

    def test_zero_concentration(self):
        mw = dna_molecular_weight(10, equal_dna_fractions)
        result = dna_weight_concentration_from_concentration("0 mol/l", 10, equal_dna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertAlmostEqual(result[0], 0.0, places=12)
        else:
            self.assertAlmostEqual(result, 0.0, places=12)

    def test_ndarray_input(self):
        mw = dna_molecular_weight(2, equal_dna_fractions)
        arr = np.array(["1 uM", "3 uM"])
        result = dna_weight_concentration_from_concentration(arr, 2, equal_dna_fractions)
        if isinstance(result, np.ndarray):
            self.assertEqual(result.shape, (2,))
            self.assertAlmostEqual(result[0], 1e-6 * mw, places=12)
            self.assertAlmostEqual(result[1], 3e-6 * mw, places=12)
        elif isinstance(result, list):
            self.assertEqual(len(result), 2)
            self.assertAlmostEqual(result[0], 1e-6 * mw, places=12)
            self.assertAlmostEqual(result[1], 3e-6 * mw, places=12)
        else:
            self.fail("Expected ndarray or list output")

class TestDNAMolesToGrams(unittest.TestCase):
    def test_scalar(self):
        from UliEngineering.Chemistry.DNA import dna_moles_to_grams, equal_dna_fractions, dna_molecular_weight
        mw = dna_molecular_weight(5, equal_dna_fractions)
        grams = dna_moles_to_grams(2, 5, equal_dna_fractions)
        self.assertAlmostEqual(grams, 2 * mw, places=8)

    def test_list(self):
        from UliEngineering.Chemistry.DNA import dna_moles_to_grams, equal_dna_fractions, dna_molecular_weight
        mw = dna_molecular_weight(3, equal_dna_fractions)
        grams = dna_moles_to_grams([1, 2], 3, equal_dna_fractions)
        self.assertEqual(len(grams), 2)
        self.assertAlmostEqual(grams[0], 1 * mw, places=8)
        self.assertAlmostEqual(grams[1], 2 * mw, places=8)

    def test_ndarray(self):
        import numpy as np
        from UliEngineering.Chemistry.DNA import dna_moles_to_grams, equal_dna_fractions, dna_molecular_weight
        mw = dna_molecular_weight(4, equal_dna_fractions)
        arr = np.array([0.5, 1.5])
        grams = dna_moles_to_grams(arr, 4, equal_dna_fractions)
        self.assertTrue(isinstance(grams, np.ndarray))
        self.assertAlmostEqual(grams[0], 0.5 * mw, places=8)
        self.assertAlmostEqual(grams[1], 1.5 * mw, places=8)

if __name__ == "__main__":
    unittest.main()

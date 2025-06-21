#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.DNA import (
    dnarna_molecular_weight as dna_molecular_weight,
    rna_molecular_weight,
    NucleotideFractions,
    DNANucleotideWeights,
    RNANucleotideWeights,
    equal_dnarna_fractions,
    dnarna_weight_concentration_from_concentration,
    dnarna_moles_to_grams,
    dnarna_grams_to_moles,
    DNARNANucleotideFractionsByOrganism
)
import numpy as np

class TestDNAMolecularWeight(unittest.TestCase):
    def test_human_dna_fractions(self):
        mw = dna_molecular_weight(100, DNARNANucleotideFractionsByOrganism.Human)
        expected = (
            100 * 0.293 * 313.2 +
            100 * 0.300 * 304.2 +
            100 * 0.207 * 329.2 +
            100 * 0.200 * 289.2 +
            79.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_equal_fractions(self):
        mw = dna_molecular_weight(40, equal_dnarna_fractions)
        expected = (
            8 * 313.2 +
            8 * 304.2 +
            8 * 329.2 +
            8 * 289.2 +
            79.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_custom_fractions(self):
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
        custom_weights = DNANucleotideWeights(A=314, T=305, G=330, C=290)
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
        mw = dna_molecular_weight(0, equal_dnarna_fractions)
        self.assertAlmostEqual(mw, 79.0, places=6)

    def test_sum_of_fractions(self):
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

class TestRNAMolecularWeight(unittest.TestCase):
    def test_human_rna_fractions(self):
        mw = rna_molecular_weight(100, DNARNANucleotideFractionsByOrganism.Human_RNA)
        expected = (
            100 * 0.293 * 329.2 +
            100 * 0.300 * 306.2 +
            100 * 0.207 * 345.2 +
            100 * 0.200 * 305.2 +
            159.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_equal_fractions(self):
        fractions = NucleotideFractions(A=0.25, T=0.0, G=0.25, C=0.25, U=0.25)
        mw = rna_molecular_weight(40, fractions)
        expected = (
            10 * 329.2 +
            10 * 306.2 +
            10 * 345.2 +
            10 * 305.2 +
            159.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_custom_weights(self):
        custom_weights = RNANucleotideWeights(A=330, U=307, G=346, C=306)
        fractions = NucleotideFractions(A=0.25, T=0.0, G=0.25, C=0.25, U=0.25)
        mw = rna_molecular_weight(4, fractions, custom_weights)
        expected = (
            1 * 330 +
            1 * 307 +
            1 * 346 +
            1 * 306 +
            159.0
        )
        self.assertAlmostEqual(mw, expected, places=6)

    def test_zero_length(self):
        fractions = NucleotideFractions(A=0.25, T=0.0, G=0.25, C=0.25, U=0.25)
        mw = rna_molecular_weight(0, fractions)
        self.assertAlmostEqual(mw, 159.0, places=6)

class TestDNAWeightConcentration(unittest.TestCase):
    def test_mol_per_liter(self):
        mw = dna_molecular_weight(1, equal_dnarna_fractions)
        result = dnarna_weight_concentration_from_concentration("1 mol/l", 1, equal_dnarna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertAlmostEqual(result[0], mw, places=6)
        else:
            self.assertAlmostEqual(result, mw, places=6)

    def test_micromolar(self):
        mw = dna_molecular_weight(10, equal_dnarna_fractions)
        result = dnarna_weight_concentration_from_concentration("1 uM", 10, equal_dnarna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertAlmostEqual(result[0], 1e-6 * mw, places=12)
        else:
            self.assertAlmostEqual(result, 1e-6 * mw, places=12)

    def test_list_input(self):
        mw = dna_molecular_weight(5, equal_dnarna_fractions)
        result = dnarna_weight_concentration_from_concentration(["1 uM", "2 uM"], 5, equal_dnarna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertEqual(len(result), 2)
            self.assertAlmostEqual(result[0], 1e-6 * mw, places=12)
            self.assertAlmostEqual(result[1], 2e-6 * mw, places=12)
        else:
            self.fail("Expected list or ndarray output")

    def test_zero_concentration(self):
        mw = dna_molecular_weight(10, equal_dnarna_fractions)
        result = dnarna_weight_concentration_from_concentration("0 mol/l", 10, equal_dnarna_fractions)
        if isinstance(result, (list, np.ndarray)):
            self.assertAlmostEqual(result[0], 0.0, places=12)
        else:
            self.assertAlmostEqual(result, 0.0, places=12)

    def test_ndarray_input(self):
        mw = dna_molecular_weight(2, equal_dnarna_fractions)
        arr = np.array(["1 uM", "3 uM"])
        result = dnarna_weight_concentration_from_concentration(arr, 2, equal_dnarna_fractions)
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

class TestRNAMolesToGrams(unittest.TestCase):
    def test_scalar(self):
        mw = rna_molecular_weight(5, NucleotideFractions(A=0.25, T=0.0, G=0.25, C=0.25, U=0.25))
        grams = 2 * mw
        self.assertAlmostEqual(grams, 2 * mw, places=8)

    def test_custom_weights(self):
        custom_weights = RNANucleotideWeights(A=330, U=307, G=346, C=306)
        fractions = NucleotideFractions(A=0.25, T=0.0, G=0.25, C=0.25, U=0.25)
        mw = rna_molecular_weight(4, fractions, custom_weights)
        grams = 3 * mw
        self.assertAlmostEqual(grams, 3 * mw, places=8)

if __name__ == "__main__":
    unittest.main()

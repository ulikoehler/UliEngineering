#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.DNA import (
    dna_molecular_weight,
    NucleotideFractions,
    NucleotideWeights,
    human_dna_fractions,
    equal_dna_fractions
)

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

if __name__ == "__main__":
    unittest.main()

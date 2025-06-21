#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Concentration import normalize_amount_concentration
from dataclasses import dataclass

__all__ = [
    "dna_molecular_weight",
    "NucleotideFractions",
    "NucleotideWeights",
    "NucleotideFractionsByOrganism",
    "equal_dna_fractions",
    "dna_weight_concentration_from_concentration",
    "dna_moles_to_grams",
    "NucleotideFractionsByOrganism"
]

@dataclass
class NucleotideWeights:
    """
    Molecular weights of the four nucleotides in g/mol.
    Source:
    https://www.thermofisher.com/de/de/home/references/ambion-tech-support/rna-tools-and-calculators/dna-and-rna-molecular-weights-and-conversions.html
    """
    A: float = 313.2   # Adenine
    T: float = 304.2   # Thymine
    G: float = 329.2   # Guanine
    C: float = 289.2   # Cytosine

@dataclass
class NucleotideFractions:
    """
    Fractions of the four nucleotides (must sum to 1.0).
    """
    A: float
    T: float
    G: float
    C: float

# Default nucleotide fractions for equal base composition
equal_dna_fractions = NucleotideFractions(A=0.25, T=0.25, G=0.25, C=0.25)

# DNA nucleotide fractions for various organisms (from HTML table data)
class NucleotideFractionsByOrganism:
    """
    Extracted from https://en.wikipedia.org/wiki/Chargaff%27s_rules
    Source:  Bansal M (2003). "DNA structure: Revisiting the Watson-Crick double helix" (PDF). Current Science. 85 (11)
    """
    Maize = NucleotideFractions(A=0.268, T=0.272, G=0.228, C=0.232)
    Octopus = NucleotideFractions(A=0.332, T=0.316, G=0.176, C=0.176)
    Chicken = NucleotideFractions(A=0.280, T=0.284, G=0.220, C=0.216)
    Rat = NucleotideFractions(A=0.286, T=0.284, G=0.214, C=0.205)
    Human = NucleotideFractions(A=0.293, T=0.300, G=0.207, C=0.200)
    Grasshopper = NucleotideFractions(A=0.293, T=0.293, G=0.205, C=0.207)
    SeaUrchin = NucleotideFractions(A=0.328, T=0.321, G=0.177, C=0.173)
    Wheat = NucleotideFractions(A=0.273, T=0.271, G=0.227, C=0.228)
    Yeast = NucleotideFractions(A=0.313, T=0.329, G=0.187, C=0.171)
    EColi = NucleotideFractions(A=0.247, T=0.236, G=0.260, C=0.257)
    PhiX174 = NucleotideFractions(A=0.240, T=0.312, G=0.233, C=0.215)

@returns_unit("g/mol")
def dna_molecular_weight(length_nucleotides, fractions: NucleotideFractions = NucleotideFractionsByOrganism.Human, nucleotide_weights: NucleotideWeights = NucleotideWeights()):
    """
    Compute the molecular weight of a single-stranded DNA molecule (e.g., oligonucleotide).

    Formula (for single-stranded DNA):
        M.W. = (An x 313.2) + (Tn x 304.2) + (Cn x 289.2) + (Gn x 329.2) + 79.0

    Parameters:
    - length_nt: Length of the DNA in nucleotides (not base pairs!).
    - fractions: NucleotideFractions dataclass instance (A, T, G, C fractions, should sum to 1.0).
    - nucleotide_weights: NucleotideWeights dataclass instance (optional, default standard values)

    Returns:
    - Molecular weight in g/mol.

    The sum of all fractions should be 1.0.

    Reference:
    https://www.thermofisher.com/de/de/home/life-science/oligonucleotides-primers-probes-genes/oligo-resource-center/oligo-basics/oligonucleotide-properties.html
    """
    length_nucleotides = normalize_numeric(length_nucleotides)
    n_A = length_nucleotides * fractions.A
    n_T = length_nucleotides * fractions.T
    n_G = length_nucleotides * fractions.G
    n_C = length_nucleotides * fractions.C

    mw = (
        n_A * nucleotide_weights.A +
        n_T * nucleotide_weights.T +
        n_G * nucleotide_weights.G +
        n_C * nucleotide_weights.C +
        79.0
    )
    return mw

def dna_weight_concentration_from_concentration(concentration, length_nucleotides, fractions: NucleotideFractions = NucleotideFractionsByOrganism.Human, nucleotide_weights: NucleotideWeights = NucleotideWeights()):
    """
    Convert DNA concentration (e.g., '5 uM', '2 mmol/l', '0.1 mol/l') to weight concentration (g/L).
    Handles scalar, list, or ndarray input.
    """
    molar_conc = normalize_amount_concentration(concentration)  # normalize to mol/L
    mw = dna_molecular_weight(length_nucleotides, fractions, nucleotide_weights)
    return molar_conc * mw  # g/L

def dna_moles_to_grams(moles, length_nucleotides, fractions: NucleotideFractions = NucleotideFractionsByOrganism.Human, nucleotide_weights: NucleotideWeights = NucleotideWeights()):
    """
    Convert amount of DNA (in moles) to grams for a given sequence length and nucleotide composition.
    Handles scalar, list, or ndarray input.
    """
    moles = normalize_numeric(moles)
    mw = dna_molecular_weight(length_nucleotides, fractions, nucleotide_weights)
    return moles * mw  # grams


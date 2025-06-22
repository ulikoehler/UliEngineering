#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Concentration import normalize_amount_concentration
from dataclasses import dataclass

__all__ = [
    "dnarna_molecular_weight",
    "rna_molecular_weight",
    "NucleotideFractions",
    "DNANucleotideWeights",
    "RNANucleotideWeights",
    "equal_dna_fractions",
    "equal_rna_fractions",
    "dnarna_weight_concentration_from_concentration",
    "dnarna_moles_to_grams",
    "dnarna_grams_to_moles",
    "DNARNANucleotideFractionsByOrganism"
]

@dataclass
class DNANucleotideWeights:
    """
    Molecular weights of the four DNA nucleotides in g/mol.
    
    This includes the phosphate group and the deoxyribose sugar.
    
    Source: https://www.thermofisher.com/de/de/home/references/ambion-tech-support/rna-tools-and-calculators/dna-and-rna-molecular-weights-and-conversions.html
    """
    A: float = 313.2   # Adenine
    T: float = 304.2   # Thymine
    G: float = 329.2   # Guanine
    C: float = 289.2   # Cytosine

@dataclass
class RNANucleotideWeights:
    """
    Molecular weights of the four RNA nucleotides in g/mol.
    This includes the backbone phosphate group.
    
    Source: https://www.thermofisher.com/de/de/home/references/ambion-tech-support/rna-tools-and-calculators/dna-and-rna-molecular-weights-and-conversions.html
    """
    A: float = 329.2   # Adenine
    U: float = 306.2   # Uracil
    G: float = 345.2   # Guanine
    C: float = 305.2   # Cytosine
    

@dataclass
class NucleotideFractions:
    """
    Fractions of the five nucleotides (must sum to 1.0).
    """
    A: float
    G: float
    C: float
    T: float = 0.0  # Default to 0 for RNA
    U: float = 0.0  # Default to 0 for DNA

# Default nucleotide fractions for equal base composition
# For DNA: U=0, for RNA: T=0
# User can override as needed

equal_dna_fractions = NucleotideFractions(A=0.25, T=0.25, G=0.25, C=0.25, U=0.0)
equal_rna_fractions = NucleotideFractions(A=0.25, T=0.0, G=0.25, C=0.25, U=0.25)

# DNA/RNA nucleotide fractions for various organisms (from HTML table data)
class DNARNANucleotideFractionsByOrganism:
    """
    Extracted from https://en.wikipedia.org/wiki/Chargaff%27s_rules
    Source:  Bansal M (2003). "DNA structure: Revisiting the Watson-Crick double helix" (PDF). Current Science. 85 (11)
    """
    Maize = NucleotideFractions(A=0.268, T=0.272, G=0.228, C=0.232, U=0.0)
    Octopus = NucleotideFractions(A=0.332, T=0.316, G=0.176, C=0.176, U=0.0)
    Chicken = NucleotideFractions(A=0.280, T=0.284, G=0.220, C=0.216, U=0.0)
    Rat = NucleotideFractions(A=0.286, T=0.284, G=0.214, C=0.205, U=0.0)
    Human = NucleotideFractions(A=0.293, T=0.300, G=0.207, C=0.200, U=0.0)
    Grasshopper = NucleotideFractions(A=0.293, T=0.293, G=0.205, C=0.207, U=0.0)
    SeaUrchin = NucleotideFractions(A=0.328, T=0.321, G=0.177, C=0.173, U=0.0)
    Wheat = NucleotideFractions(A=0.273, T=0.271, G=0.227, C=0.228, U=0.0)
    Yeast = NucleotideFractions(A=0.313, T=0.329, G=0.187, C=0.171, U=0.0)
    EColi = NucleotideFractions(A=0.247, T=0.236, G=0.260, C=0.257, U=0.0)
    PhiX174 = NucleotideFractions(A=0.240, T=0.312, G=0.233, C=0.215, U=0.0)
    # Example for RNA (T=0, U>0):
    Human_RNA = NucleotideFractions(A=0.293, T=0.0, G=0.207, C=0.200, U=0.300)

@returns_unit("g/mol")
def dnarna_molecular_weight(length_nucleotides, fractions: NucleotideFractions = DNARNANucleotideFractionsByOrganism.Human, nucleotide_weights: DNANucleotideWeights = DNANucleotideWeights()):
    """
    Compute the molecular weight of a single-stranded DNA molecule (e.g., oligonucleotide).

    Formula (for single-stranded DNA):
        M.W. = (An x 313.2) + (Tn x 304.2) + (Cn x 289.2) + (Gn x 329.2) + 79.0

    Parameters:
    - length_nucleotides: Length of the DNA in nucleotides (not base pairs!).
    - fractions: NucleotideFractions dataclass instance (A, T, G, C, U fractions, should sum to 1.0, U ignored).
    - nucleotide_weights: DNANucleotideWeights dataclass instance (optional, default standard values)

    Returns:
    - Molecular weight in g/mol.
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

@returns_unit("g/mol")
def rna_molecular_weight(length_nucleotides, fractions: NucleotideFractions = DNARNANucleotideFractionsByOrganism.Human_RNA, nucleotide_weights: RNANucleotideWeights = RNANucleotideWeights()):
    """
    Compute the molecular weight of a single-stranded RNA molecule (e.g., oligonucleotide).

    Formula (for single-stranded RNA):
        M.W. = (An x 329.2) + (Un x 306.2) + (Cn x 305.2) + (Gn x 345.2) + 159

    Parameters:
    - length_nucleotides: Length of the RNA in nucleotides (not base pairs!).
    - fractions: NucleotideFractions dataclass instance (A, U, G, C, T fractions, should sum to 1.0, T ignored).
    - nucleotide_weights: RNANucleotideWeights dataclass instance (optional, default standard values)

    Returns:
    - Molecular weight in g/mol.
    """
    length_nucleotides = normalize_numeric(length_nucleotides)
    n_A = length_nucleotides * fractions.A
    n_U = length_nucleotides * getattr(fractions, 'U', 0.0)
    n_G = length_nucleotides * fractions.G
    n_C = length_nucleotides * fractions.C
    mw = (
        n_A * nucleotide_weights.A +
        n_U * nucleotide_weights.U +
        n_G * nucleotide_weights.G +
        n_C * nucleotide_weights.C +
        159.0
    )
    return mw

@returns_unit("g/L")
def dnarna_weight_concentration_from_concentration(concentration, length_nucleotides, fractions: NucleotideFractions = DNARNANucleotideFractionsByOrganism.Human, nucleotide_weights: DNANucleotideWeights = DNANucleotideWeights()):
    """
    Convert DNA/RNA concentration (e.g., '5 uM', '2 mmol/l', '0.1 mol/l') to weight concentration (g/L).
    Handles scalar, list, or ndarray input.
    """
    molar_conc = normalize_amount_concentration(concentration)  # normalize to mol/L
    mw = dnarna_molecular_weight(length_nucleotides, fractions, nucleotide_weights)
    return molar_conc * mw  # g/L

@returns_unit("g")
def dnarna_moles_to_grams(moles, length_nucleotides, fractions: NucleotideFractions = DNARNANucleotideFractionsByOrganism.Human, nucleotide_weights: DNANucleotideWeights = DNANucleotideWeights()):
    """
    Convert amount of DNA/RNA (in moles) to grams for a given sequence length and nucleotide composition.
    Handles scalar, list, or ndarray input.
    """
    moles = normalize_numeric(moles)
    mw = dnarna_molecular_weight(length_nucleotides, fractions, nucleotide_weights)
    return moles * mw  # grams

@returns_unit("mol")
def dnarna_grams_to_moles(grams, length_nucleotides, fractions: NucleotideFractions = DNARNANucleotideFractionsByOrganism.Human, nucleotide_weights: DNANucleotideWeights = DNANucleotideWeights()):
    """
    Convert mass of DNA/RNA (in grams) to moles for a given sequence length and nucleotide composition.
    Handles scalar, list, or ndarray input.
    """
    grams = normalize_numeric(grams)
    mw = dnarna_molecular_weight(length_nucleotides, fractions, nucleotide_weights)
    return grams / mw  # moles


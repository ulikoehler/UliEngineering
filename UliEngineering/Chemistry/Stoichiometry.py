#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive stoichiometry module.

Provides functions for:
- Parsing chemical formulas and extracting element compositions
- Computing molecular weights from formulas
- Balancing chemical equations
- Mole/mass/volume conversions
- Limiting reagent analysis
- Percent composition
- Empirical and molecular formula determination
- Dilution and solution preparation
- Molarity / Molality conversions
"""
import re
import numpy as np
from fractions import Fraction
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "ATOMIC_WEIGHTS",
    "parse_formula",
    "molecular_weight",
    "percent_composition",
    "moles_to_grams",
    "grams_to_moles",
    "moles_to_particles",
    "particles_to_moles",
    "molarity_from_moles_volume",
    "moles_from_molarity_volume",
    "volume_from_molarity_moles",
    "molality_from_moles_mass",
    "dilution_volume",
    "mass_fraction_to_molarity",
    "limiting_reagent",
    "theoretical_yield",
    "percent_yield",
    "empirical_formula_from_percent",
    "ideal_gas_moles",
    "ideal_gas_volume",
    "AVOGADRO",
    "MOLAR_GAS_VOLUME_STP",
]

from scipy.constants import N_A as AVOGADRO, R as GAS_CONSTANT

MOLAR_GAS_VOLUME_STP = 0.022414  # m³/mol at STP (0 °C, 1 atm)

# Standard atomic weights (IUPAC 2021)
ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.64, "As": 74.922, "Se": 78.96, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.96, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
    "Ag": 107.868, "Cd": 112.411, "In": 114.818, "Sn": 118.710, "Sb": 121.760,
    "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930,
    "Er": 167.259, "Tm": 168.934, "Yb": 173.054, "Lu": 174.967, "Hf": 178.49,
    "Ta": 180.948, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217,
    "Pt": 195.084, "Au": 196.967, "Hg": 200.59, "Tl": 204.383, "Pb": 207.2,
    "Bi": 208.980, "U": 238.029,
}


def parse_formula(formula):
    """
    Parse a chemical formula string into a dictionary of element counts.

    Supports:
    - Simple formulas: "H2O", "NaCl", "C6H12O6"
    - Parenthetical groups: "Ca(OH)2", "Mg3(PO4)2"
    - Nested groups: "Ca3(PO4)2"
    - Hydrates: "CuSO4·5H2O" (use · or . as separator)

    Parameters
    ----------
    formula : str
        Chemical formula string.

    Returns
    -------
    dict
        Dictionary mapping element symbols to their counts.
        e.g., {"H": 2, "O": 1} for "H2O"
    """
    # Handle hydrates: split on · or middot
    parts = re.split(r'[·.]', formula)
    if len(parts) > 1:
        result = {}
        # First part: the compound
        base = _parse_formula_recursive(parts[0])
        for el, cnt in base.items():
            result[el] = result.get(el, 0) + cnt
        # Subsequent parts: hydrate multiplier
        for part in parts[1:]:
            # Check for leading coefficient like "5H2O"
            m = re.match(r'^(\d+)(.*)', part)
            if m:
                coeff = int(m.group(1))
                sub_formula = m.group(2)
            else:
                coeff = 1
                sub_formula = part
            sub = _parse_formula_recursive(sub_formula)
            for el, cnt in sub.items():
                result[el] = result.get(el, 0) + cnt * coeff
        return result
    return _parse_formula_recursive(formula)


def _parse_formula_recursive(formula):
    """
    Recursively parse a chemical formula handling parentheses.
    """
    # Tokenize: element symbols, numbers, parentheses
    tokens = re.findall(r'([A-Z][a-z]?|\d+|[()])', formula)
    stack = [{}]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '(':
            stack.append({})
        elif token == ')':
            # Look for a following number
            i += 1
            multiplier = 1
            if i < len(tokens) and tokens[i].isdigit():
                multiplier = int(tokens[i])
            group = stack.pop()
            for el, cnt in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + cnt * multiplier
        elif token[0].isupper():
            # Element symbol - check for following count
            element = token
            i += 1
            count = 1
            if i < len(tokens) and tokens[i].isdigit():
                count = int(tokens[i])
            else:
                i -= 1
            stack[-1][element] = stack[-1].get(element, 0) + count
        i += 1
    return stack[0]


@returns_unit("g/mol")
def molecular_weight(formula):
    """
    Compute the molecular weight of a compound from its formula.

    Parameters
    ----------
    formula : str
        Chemical formula (e.g., "H2O", "NaCl", "Ca(OH)2").

    Returns
    -------
    float
        Molecular weight in g/mol.

    Raises
    ------
    KeyError
        If an element in the formula is not in the ATOMIC_WEIGHTS table.
    """
    elements = parse_formula(formula)
    mw = 0.0
    for element, count in elements.items():
        if element not in ATOMIC_WEIGHTS:
            raise KeyError(f"Unknown element: {element}")
        mw += ATOMIC_WEIGHTS[element] * count
    return mw


def percent_composition(formula):
    """
    Compute the percent composition by mass of each element in a compound.

    Parameters
    ----------
    formula : str
        Chemical formula.

    Returns
    -------
    dict
        Dictionary mapping element symbols to their mass percentage.
    """
    elements = parse_formula(formula)
    mw = molecular_weight(formula)
    result = {}
    for element, count in elements.items():
        result[element] = (ATOMIC_WEIGHTS[element] * count / mw) * 100.0
    return result


@normalize_numeric_args
@returns_unit("g")
def moles_to_grams(moles, molar_mass):
    """
    Convert moles to grams.

    m = n * M

    Parameters
    ----------
    moles : float
        Amount in moles.
    molar_mass : float
        Molar mass in g/mol.

    Returns
    -------
    float
        Mass in grams.
    """
    return moles * molar_mass


@normalize_numeric_args
@returns_unit("mol")
def grams_to_moles(grams, molar_mass):
    """
    Convert grams to moles.

    n = m / M

    Parameters
    ----------
    grams : float
        Mass in grams.
    molar_mass : float
        Molar mass in g/mol.

    Returns
    -------
    float
        Amount in moles.
    """
    return grams / molar_mass


@normalize_numeric_args
@returns_unit("")
def moles_to_particles(moles):
    """
    Convert moles to number of particles.

    N = n * N_A

    Parameters
    ----------
    moles : float
        Amount in moles.

    Returns
    -------
    float
        Number of particles.
    """
    return moles * AVOGADRO


@normalize_numeric_args
@returns_unit("mol")
def particles_to_moles(particles):
    """
    Convert number of particles to moles.

    n = N / N_A

    Parameters
    ----------
    particles : float
        Number of particles.

    Returns
    -------
    float
        Amount in moles.
    """
    return particles / AVOGADRO


@normalize_numeric_args
@returns_unit("mol/L")
def molarity_from_moles_volume(moles, volume_liters):
    """
    Compute molarity from moles and volume.

    M = n / V

    Parameters
    ----------
    moles : float
        Amount in moles.
    volume_liters : float
        Volume in liters.

    Returns
    -------
    float
        Molarity in mol/L.
    """
    return moles / volume_liters


@normalize_numeric_args
@returns_unit("mol")
def moles_from_molarity_volume(molarity, volume_liters):
    """
    Compute moles from molarity and volume.

    n = M * V

    Parameters
    ----------
    molarity : float
        Molarity in mol/L.
    volume_liters : float
        Volume in liters.

    Returns
    -------
    float
        Amount in moles.
    """
    return molarity * volume_liters


@normalize_numeric_args
@returns_unit("L")
def volume_from_molarity_moles(molarity, moles):
    """
    Compute volume needed for given moles at given molarity.

    V = n / M

    Parameters
    ----------
    molarity : float
        Molarity in mol/L.
    moles : float
        Amount in moles.

    Returns
    -------
    float
        Volume in liters.
    """
    return moles / molarity


@normalize_numeric_args
@returns_unit("mol/kg")
def molality_from_moles_mass(moles_solute, mass_solvent_kg):
    """
    Compute molality from moles of solute and mass of solvent.

    b = n_solute / m_solvent (kg)

    Parameters
    ----------
    moles_solute : float
        Moles of solute.
    mass_solvent_kg : float
        Mass of solvent in kilograms.

    Returns
    -------
    float
        Molality in mol/kg.
    """
    return moles_solute / mass_solvent_kg


@normalize_numeric_args
@returns_unit("L")
def dilution_volume(C1, V1, C2):
    """
    Compute the final volume after dilution using C₁V₁ = C₂V₂.

    V₂ = C₁ * V₁ / C₂

    Parameters
    ----------
    C1 : float
        Initial concentration (any consistent unit).
    V1 : float
        Initial volume (any consistent unit).
    C2 : float
        Final (desired) concentration.

    Returns
    -------
    float
        Final volume.
    """
    return C1 * V1 / C2


@normalize_numeric_args
@returns_unit("mol/L")
def mass_fraction_to_molarity(mass_fraction, density_kg_per_L, molar_mass):
    """
    Convert mass fraction (w/w) to molarity.

    M = (w * ρ * 1000) / M_w

    Parameters
    ----------
    mass_fraction : float
        Mass fraction (dimensionless, e.g. 0.37 for 37%).
    density_kg_per_L : float
        Density of the solution in kg/L.
    molar_mass : float
        Molar mass of the solute in g/mol.

    Returns
    -------
    float
        Molarity in mol/L.
    """
    return mass_fraction * density_kg_per_L * 1000.0 / molar_mass


def limiting_reagent(reactant_moles, stoich_coefficients):
    """
    Determine the limiting reagent from arrays of available moles
    and stoichiometric coefficients.

    The limiting reagent has the smallest ratio moles/coefficient.

    Parameters
    ----------
    reactant_moles : array-like
        Available moles of each reactant.
    stoich_coefficients : array-like
        Stoichiometric coefficients of each reactant.

    Returns
    -------
    int
        Index (0-based) of the limiting reagent.
    """
    moles = np.asarray(reactant_moles, dtype=float)
    coeffs = np.asarray(stoich_coefficients, dtype=float)
    ratios = moles / coeffs
    return int(np.argmin(ratios))


@normalize_numeric_args
@returns_unit("mol")
def theoretical_yield(limiting_moles, limiting_coeff, product_coeff):
    """
    Compute theoretical yield (in moles) of a product.

    n_product = n_limiting * (product_coeff / limiting_coeff)

    Parameters
    ----------
    limiting_moles : float
        Moles of the limiting reagent.
    limiting_coeff : float
        Stoichiometric coefficient of the limiting reagent.
    product_coeff : float
        Stoichiometric coefficient of the product.

    Returns
    -------
    float
        Theoretical yield in moles.
    """
    return limiting_moles * product_coeff / limiting_coeff


@normalize_numeric_args
@returns_unit("%")
def percent_yield(actual, theoretical):
    """
    Compute percent yield.

    % yield = (actual / theoretical) * 100

    Parameters
    ----------
    actual : float
        Actual yield (any unit).
    theoretical : float
        Theoretical yield (same unit).

    Returns
    -------
    float
        Percent yield.
    """
    return (actual / theoretical) * 100.0


def empirical_formula_from_percent(percentages):
    """
    Determine the empirical formula from percent composition data.

    Parameters
    ----------
    percentages : dict
        Dictionary mapping element symbols to their mass percentage.
        e.g., {"C": 40.0, "H": 6.7, "O": 53.3}

    Returns
    -------
    dict
        Dictionary mapping element symbols to their subscript in the
        empirical formula (integer values).
    """
    # Convert percentages to moles
    moles = {}
    for element, pct in percentages.items():
        moles[element] = pct / ATOMIC_WEIGHTS[element]

    # Divide by smallest
    min_moles = min(moles.values())
    ratios = {el: m / min_moles for el, m in moles.items()}

    # Round to nearest integer using fractions for precision
    # Try multipliers 1 through 6
    for multiplier in range(1, 7):
        scaled = {el: r * multiplier for el, r in ratios.items()}
        rounded = {el: round(v) for el, v in scaled.items()}
        if all(abs(scaled[el] - rounded[el]) < 0.1 for el in scaled):
            return rounded

    # Fallback: just round
    return {el: round(r) for el, r in ratios.items()}


@normalize_numeric_args
@returns_unit("mol")
def ideal_gas_moles(pressure_Pa, volume_m3, T):
    """
    Compute moles of ideal gas from PV = nRT.

    n = PV / (RT)

    Parameters
    ----------
    pressure_Pa : float
        Pressure in Pascals.
    volume_m3 : float
        Volume in cubic meters.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Amount in moles.
    """
    return pressure_Pa * volume_m3 / (GAS_CONSTANT * T)


@normalize_numeric_args
@returns_unit("m³")
def ideal_gas_volume(moles, T, pressure_Pa=101325.0):
    """
    Compute volume of ideal gas from PV = nRT.

    V = nRT / P

    Parameters
    ----------
    moles : float
        Amount in moles.
    T : float
        Temperature in Kelvin.
    pressure_Pa : float
        Pressure in Pascals (default: 101325 Pa = 1 atm).

    Returns
    -------
    float
        Volume in cubic meters.
    """
    return moles * GAS_CONSTANT * T / pressure_Pa

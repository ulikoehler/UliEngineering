#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ionic strength calculations.

Ionic strength I is a measure of the total concentration of ions in solution:

    I = 0.5 * Σ(c_i * z_i²)

where c_i is the molar concentration and z_i is the charge number of ion i.
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
import numpy as np

__all__ = [
    "ionic_strength",
    "ionic_strength_from_pairs",
    "ionic_strength_monovalent",
    "normalize_concentration", "ConcentrationMolar",
]


def normalize_concentration(c: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(c, {"mol/L": 1.0, "M": 1.0, "mM": 1e-3, "µM": 1e-6, "mol/m³": 1e-3}, quantity_name="concentration")

ConcentrationMolar = Annotated[NormalizedComputable, normalize_concentration]


@returns_unit("mol/L")
def ionic_strength(concentrations, charges):
    """
    Compute the ionic strength from arrays of concentrations and charges.

    I = 0.5 * Σ(c_i * z_i²)

    Parameters
    ----------
    concentrations : array-like
        Molar concentrations of each ion in mol/L.
    charges : array-like
        Charge numbers of each ion (signed integers, e.g., +1, -1, +2, -2).

    Returns
    -------
    float
        Ionic strength in mol/L.
    """
    c = np.asarray(concentrations, dtype=float)
    z = np.asarray(charges, dtype=float)
    return 0.5 * np.sum(c * z**2)


@returns_unit("mol/L")
def ionic_strength_from_pairs(pairs):
    """
    Compute ionic strength from a list of (concentration, charge) tuples.

    Parameters
    ----------
    pairs : list of (float, int) tuples
        Each tuple is (concentration_in_mol_per_L, charge_number).

    Returns
    -------
    float
        Ionic strength in mol/L.
    """
    concentrations = [p[0] for p in pairs]
    charges = [p[1] for p in pairs]
    return ionic_strength(concentrations, charges)


@returns_unit("mol/L")
def ionic_strength_monovalent(concentration: ConcentrationMolar):
    """
    Compute ionic strength for a monovalent salt (e.g. NaCl).
    For a 1:1 electrolyte MX at concentration c, I = c.

    Parameters
    ----------
    concentration : float
        Molar concentration of the salt in mol/L.

    Returns
    -------
    float
        Ionic strength in mol/L (equals the concentration for 1:1 salts).
    """
    concentration = normalize_concentration(concentration) if isinstance(concentration, str) else concentration
    return float(concentration)

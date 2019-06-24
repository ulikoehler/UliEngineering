#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing tolerances
"""
from UliEngineering.Units import Unit
from UliEngineering.EngineerIO import normalize
from UliEngineering.Utils.Range import normalize_minmax_tuple, ValueRange
from collections import namedtuple

__all__ = ["value_range_over_tolerance"]

def value_range_over_tolerance(nominal, tolerance="1 %"):
    """
    Compute the minimum and maximum value of a given component,
    given its nominal value and its tolerance.
    """
    normalized = normalize(nominal)
    nominal, unit = normalized.value, normalized.unit
    # Parse static tolerance
    min_tol_coeff, max_tol_coeff, nix = normalize_minmax_tuple(tolerance, name="tolerance")
    tol_neg_factor = 1. + min_tol_coeff
    tol_pos_factor = 1. + max_tol_coeff
    return ValueRange(tol_neg_factor * nominal, tol_pos_factor * nominal, unit)

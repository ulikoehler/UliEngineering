#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for mass normalization."""
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable

from ._normalize import normalize_with_known_units

__all__ = ["normalize_mass_grams"]


def normalize_mass_grams(mass: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(mass, {"kg": 1000.0, "mg": 0.001, "g": 1.0}, quantity_name="mass")
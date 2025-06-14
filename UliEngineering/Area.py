#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for area
"""
from numpy import ndarray
import scipy.constants
import numpy as np
from .EngineerIO import EngineerIO, returns_unit
from .Units import UnknownUnitInContextException

__all__ = ["normalize_area", "convert_area_to_square_meters"]

_area_factors = {
    "": 1., # Assumed. It's SI!
    "m²": 1.,
    "m^2": 1.,
    "square meter": 1.,
    "square meters": 1.,
    "sq m": 1.,
    "sqm": 1.,
    'mm²': 1e-6,
    'mm^2': 1e-6,
    'square millimeter': 1e-6,
    'square millimeters': 1e-6,
    'sq mm': 1e-6,
    'cm²': 1e-4,
    'cm^2': 1e-4,
    'square centimeter': 1e-4,
    'square centimeters': 1e-4,
    'sq cm': 1e-4,
    'dm²': 1e-2,
    'dm^2': 1e-2,
    'square decimeter': 1e-2,
    'square decimeters': 1e-2,
    'sq dm': 1e-2,
    'km²': 1e6,
    'km^2': 1e6,
    'square kilometer': 1e6,
    'square kilometers': 1e6,
    'sq km': 1e6,
    'in²': scipy.constants.inch**2,
    'in^2': scipy.constants.inch**2,
    'square inch': scipy.constants.inch**2,
    'square inches': scipy.constants.inch**2,
    'sq in': scipy.constants.inch**2,
    'ft²': scipy.constants.foot**2,
    'ft^2': scipy.constants.foot**2,
    'square foot': scipy.constants.foot**2,
    'square feet': scipy.constants.foot**2,
    'sq ft': scipy.constants.foot**2,
    'yd²': scipy.constants.yard**2,
    'yd^2': scipy.constants.yard**2,
    'square yard': scipy.constants.yard**2,
    'square yards': scipy.constants.yard**2,
    'sq yd': scipy.constants.yard**2,
    'acre': 4046.8564224,
    'acres': 4046.8564224,
    'hectare': 10000.,
    'hectares': 10000.,
    'ha': 10000.,
    'are': 100.,
    'ares': 100.,
    'barn': 1e-28,
    'barns': 1e-28,
    'b': 1e-28,
}

@returns_unit("m²")
def convert_area_to_square_meters(value, unit, instance=EngineerIO.area_instance):
    """
    Given a number or Engineer string (unit ignored) <value>
    in <unit>, convert it to square meters.
    """
    # Currently a hack, but doing it directly will not parse SI units
    return normalize_area(f"{value} {unit}", instance=EngineerIO.area_instance)

@returns_unit("m²")
def normalize_area(s, instance=EngineerIO.area_instance):
    """
    Normalize an area to square meters.
    Returns the numeric value in m², a list or ndarray of converted values,
    or None if the input is None.

    Valid inputs include:
    - "1.0" => 1.0
    - "1.0 cm²" => 0.0001
    - "1 square inch" => 0.00064516
    - "1 acre" => 4046.8564224
    - "2.5 hectares" => 25000.0
    - "100 sq ft" => 9.290304
    """
    if s is None:
        return None
    if isinstance(s, list):
        return [normalize_area(v) for v in s]
    if isinstance(s, ndarray):
        return np.asarray([normalize_area(v) for v in s])
    result = instance.normalize(s)
    if result.unit in _area_factors:
        return result.value * _area_factors[result.unit]
    raise UnknownUnitInContextException(f"Unknown area unit: {result.unit}")
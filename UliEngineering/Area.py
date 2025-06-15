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
    'square mm': 1e-6,
    'sq mm': 1e-6,
    'mm sq': 1e-6,
    'mm squared': 1e-6,
    'millimeter squared': 1e-6,
    'cm²': 1e-4,
    'cm^2': 1e-4,
    'square centimeter': 1e-4,
    'square centimeters': 1e-4,
    'square cm': 1e-4,
    'sq cm': 1e-4,
    'cm sq': 1e-4,
    'cm squared': 1e-4,
    'centimeter squared': 1e-4,
    'centimeters squared': 1e-4,
    'dm²': 1e-2,
    'dm^2': 1e-2,
    'square decimeter': 1e-2,
    'square decimeters': 1e-2,
    'square dm': 1e-2,
    'sq dm': 1e-2,
    'dm sq': 1e-2,
    'dm squared': 1e-2,
    'decimeter squared': 1e-2,
    'decimeters squared': 1e-2,
    'µm²': 1e-12,
    'µm^2': 1e-12,
    'um²': 1e-12,
    'um^2': 1e-12,
    'square micrometer': 1e-12,
    'square micrometers': 1e-12,
    'square µm': 1e-12,
    'sq µm': 1e-12,
    'sq um': 1e-12,
    'um sq': 1e-12,
    'µm sq': 1e-12,
    'µm squared': 1e-12,
    'micrometer squared': 1e-12,
    'micrometers squared': 1e-12,
    'nm²': 1e-18,
    'nm^2': 1e-18,
    'square nanometer': 1e-18,
    'square nanometers': 1e-18,
    'square nm': 1e-18,
    'sq nm': 1e-18,
    'nm sq': 1e-18,
    'nm squared': 1e-18,
    'nanometer squared': 1e-18,
    'nanometers squared': 1e-18,
    'km²': 1e6,
    'km^2': 1e6,
    'square kilometer': 1e6,
    'square kilometers': 1e6,
    'square km': 1e6,
    'sq km': 1e6,
    'km squared': 1e6,
    'km sq': 1e6,
    'kilometers sq': 1e6,
    'kilometers squared': 1e6,
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
    'square µm': 1e-6,
    'sq µm': 1e-6,
    'µm sq': 1e-6,
    'µm squared': 1e-6,
    # Additional missing variations
    'µbarn': 1e-34,  # microbarn
    'mbarn': 1e-31,  # millibarn
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
    # We can't just normalize() it here
    # We need to takes care of the fact that cm² means (cm)², nm² means (nm)²
    # whereas µbarn means micro(barns), hence the multiplier does not need to be squared.
    # Heuristic: everything with square in it (², ^2 etc) must be squared
    split_result = instance.normalize(s)
    unit_lowercase = split_result.unit.lower() # "m²" etc
    is_square_unit = False
    for square_denotation in ["²", "^2", "m2", "in2", "ft2", "square", "sq ", " sq"]:
        if square_denotation in unit_lowercase:
            # We have a square unit, so we need to square the value
            is_square_unit = True
            break
    # NOTE: The value has already been multiplied by the prefix factor ONCE,
    # so we need to multiply it again (to make it squared in total)
    if is_square_unit:
        split_result.value *= split_result.prefix_multiplier
    # Multiply unit-specific factor to convert to meters square
    if split_result.unit in _area_factors:
        print(f"Converting {split_result.value} {split_result.unit} to m²")
        print(f"Using factor {_area_factors[split_result.unit]}")
        print(split_result)
        return split_result.value * _area_factors[split_result.unit]
    else:
        raise UnknownUnitInContextException(
            f"Unknown area unit '{split_result.unit}' for input string {s}!",
        )
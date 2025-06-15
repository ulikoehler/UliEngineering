#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for area
"""
from typing import Dict, Set
from numpy import ndarray
import scipy.constants
import numpy as np
from . import EngineerIO
from .Decorators import returns_unit
from ..Units import UnknownUnitInContextException

__all__ = ["normalize_area", "convert_area_to_square_meters", "EngineerAreaIO"]


def _area_units() -> Set[str]:
    """
    All known area units (compact symbols only).
    """
    units = set([
        # Area units
        'm²', 'm^2',
        'in²', 'in^2',
        'ft²', 'ft^2', 
        'yd²', 'yd^2',
        'acre', 'ha', 'are',
        'barn', 'b',
        # NOTE: Do not list SI-prefixed units such as cm² here!
        # These must be parsed as m² with SI prefix "c" etc.
    ])
    return units

def _area_unit_aliases() -> Dict[str, str]:
    """
    Maps verbose area unit names to their compact symbols.
    """
    return {
        # Square inch aliases
        'square inch': 'in²',
        'square inches': 'in²', 
        'sq in': 'in²',
        
        # Square foot aliases
        'square foot': 'ft²',
        'square feet': 'ft²', 
        'sq ft': 'ft²',
        
        # Square yard aliases  
        'square yard': 'yd²',
        'square yards': 'yd²',
        'sq yd': 'yd²',
        
        # Other area aliases
        'acres': 'acre',
        'hectare': 'ha', 
        'hectares': 'ha', 
        'hectars': 'ha', 
        'ares': 'are',
        'barns': 'barn',
        'square meter': 'm²',
        'square meters': 'm²', 
        'sq m': 'm²', 
        'sqm': 'm²',
        'm^2': 'm²',  # Caret notation alias
        
        # SI prefixed meter aliases
        'square millimeter': 'mm²',
        'square millimeters': 'mm²', 
        'square mm': 'mm²', 
        'sq mm': 'mm²', 
        'mm sq': 'mm²', 
        'mm squared': 'mm²', 
        'millimeter squared': 'mm²',
        'mm^2': 'mm²',  # Caret notation alias
        
        'square centimeter': 'cm²',
        'square centimeters': 'cm²', 
        'square cm': 'cm²', 
        'sq cm': 'cm²', 
        'cm sq': 'cm²', 
        'cm squared': 'cm²', 
        'centimeter squared': 'cm²', 
        'centimeters squared': 'cm²',
        'cm^2': 'cm²',  # Caret notation alias
        
        'square decimeter': 'dm²',
        'square decimeters': 'dm²', 
        'square dm': 'dm²', 
        'sq dm': 'dm²', 
        'dm sq': 'dm²', 
        'dm squared': 'dm²', 
        'decimeter squared': 'dm²', 
        'decimeters squared': 'dm²',
        'dm^2': 'dm²',  # Caret notation alias
        
        'square micrometer': 'µm²',
        'square micrometers': 'µm²', 
        'square µm': 'µm²',
        'sq µm': 'µm²', 
        'sq um': 'µm²', 
        'um sq': 'µm²', 
        'µm sq': 'µm²', 
        'µm squared': 'µm²', 
        'micrometer squared': 'µm²', 
        'micrometers squared': 'µm²',
        'µm^2': 'µm²',  # Caret notation alias
        'um^2': 'µm²',  # Caret notation alias
        
        'square nanometer': 'nm²',
        'square nanometers': 'nm²', 
        'square nm': 'nm²', 
        'sq nm': 'nm²', 
        'nm sq': 'nm²', 
        'nm squared': 'nm²', 
        'nanometers squared': 'nm²',
        'nm^2': 'nm²',  # Caret notation alias
        
        'square kilometer': 'km²',
        'square kilometers': 'km²', 
        'square km': 'km²', 
        'sq km': 'km²', 
        'km squared': 'km²', 
        'km sq': 'km²', 
        'kilometers sq': 'km²', 
        'kilometers squared': 'km²',
        'km^2': 'km²',  # Caret notation alias
    }

def _default_unit_prefix_map(include_length_unit_prefixes=False):
    """
    Returns the default unit prefix map for area calculations
    """
    prefix_map = {
        'y': -24, 'z': -21, 'a': -18, 'f': -15, 'p': -12,
        'n': -9, 'µ': -6, 'μ': -6, 'u': -6, 'm': -3,
        '': 0,
        'k': 3, 'M': 6, 'G': 9, 'T': 12, 'P': 15,
        'E': 18, 'Z': 21, 'Y': 24
    }
    
    if include_length_unit_prefixes:
        prefix_map.update({
            'c': -2,  # centi
            'd': -1   # deci
        })
    
    return prefix_map

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


class EngineerAreaIO(EngineerIO):
    """
    EngineerIO subclass specialized for area unit parsing and conversion.
    """
    
    _instance = None
    
    def __init__(self):
        # Use area-specific configuration
        super().__init__(
            units=_area_units(),
            unit_aliases=_area_unit_aliases(),
            unit_prefix_map=_default_unit_prefix_map(include_length_unit_prefixes=True)
        )
    
    @classmethod
    def instance(cls):
        """
        Get the singleton instance of EngineerAreaIO
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @returns_unit("m²")
    def normalize_area(self, s):
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
            return [self.normalize_area(v) for v in s]
        if isinstance(s, ndarray):
            return np.asarray([self.normalize_area(v) for v in s])
        # We can't just normalize() it here
        # We need to takes care of the fact that cm² means (cm)², nm² means (nm)²
        # whereas µbarn means micro(barns), hence the multiplier does not need to be squared.
        # Heuristic: everything with square in it (², ^2 etc) must be squared
        split_result = self.normalize(s)
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
            return split_result.value * _area_factors[split_result.unit]
        else:
            raise UnknownUnitInContextException(
                f"Unknown area unit '{split_result.unit}' for input string {s}!",
            )
    
    @returns_unit("m²")
    def convert_area_to_square_meters(self, value, unit):
        """
        Given a number or Engineer string (unit ignored) <value>
        in <unit>, convert it to square meters.
        """
        # Currently a hack, but doing it directly will not parse SI units
        return self.normalize_area(f"{value} {unit}")

def _area_units():
    """
    Returns a set of area unit symbols
    """
    return {
        "m²", "m^2",
        "mm²", "mm^2",
        "cm²", "cm^2", 
        "dm²", "dm^2",
        "km²", "km^2",
        "µm²", "µm^2", "um²", "um^2",
        "nm²", "nm^2",
        "in²", "in^2",
        "ft²", "ft^2", 
        "yd²", "yd^2",
        "acre", "ha", "are", "barn", "barns", "b"
    }

def _area_unit_aliases():
    """
    Returns a dictionary mapping area unit aliases to their canonical symbols
    """
    return {
        # Square meter aliases
        "square meter": "m²",
        "square meters": "m²", 
        "sq m": "m²",
        "sqm": "m²",
        
        # Square millimeter aliases
        "square millimeter": "mm²",
        "square millimeters": "mm²",
        "square mm": "mm²",
        "sq mm": "mm²",
        "mm sq": "mm²",
        "mm squared": "mm²",
        "millimeter squared": "mm²",
        
        # Square centimeter aliases
        "square centimeter": "cm²",
        "square centimeters": "cm²",
        "square cm": "cm²",
        "sq cm": "cm²",
        "cm sq": "cm²",
        "cm squared": "cm²",
        "centimeter squared": "cm²",
        "centimeters squared": "cm²",
        
        # Square decimeter aliases
        "square decimeter": "dm²",
        "square decimeters": "dm²",
        "square dm": "dm²",
        "sq dm": "dm²",
        "dm sq": "dm²",
        "dm squared": "dm²",
        "decimeter squared": "dm²",
        "decimeters squared": "dm²",
        
        # Square kilometer aliases
        "square kilometer": "km²",
        "square kilometers": "km²",
        "square km": "km²",
        "sq km": "km²",
        "km squared": "km²",
        "km sq": "km²",
        "kilometers sq": "km²",
        "kilometers squared": "km²",
        
        # Square micrometer aliases
        "square micrometer": "µm²",
        "square micrometers": "µm²",
        "square µm": "µm²",
        "sq µm": "µm²",
        "sq um": "µm²",
        "um sq": "µm²",
        "µm sq": "µm²",
        "µm squared": "µm²",
        "micrometer squared": "µm²",
        "micrometers squared": "µm²",
        
        # Square nanometer aliases
        "square nanometer": "nm²",
        "square nanometers": "nm²",
        "square nm": "nm²",
        "sq nm": "nm²",
        "nm sq": "nm²",
        "nm squared": "nm²",
        "nanometer squared": "nm²",
        "nanometers squared": "nm²",
        
        # Imperial units
        "square inch": "in²",
        "square inches": "in²",
        "sq in": "in²",
        "square foot": "ft²",
        "square feet": "ft²",
        "sq ft": "ft²",
        "square yard": "yd²",
        "square yards": "yd²",
        "sq yd": "yd²",
        
        # Agricultural units
        "acres": "acre",
        "hectare": "ha",
        "hectares": "ha",
        "ares": "are"
    }

# Backward compatibility functions
@returns_unit("m²")
def normalize_area(s, instance=None):
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
    if instance is None:
        instance = EngineerAreaIO.instance()
    return instance.normalize_area(s)

@returns_unit("m²")
def convert_area_to_square_meters(value, unit, instance=None):
    """
    Given a number or Engineer string (unit ignored) <value>
    in <unit>, convert it to square meters.
    """
    if instance is None:
        instance = EngineerAreaIO.instance()
    return instance.convert_area_to_square_meters(value, unit)
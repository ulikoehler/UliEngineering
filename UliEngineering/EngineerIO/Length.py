#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for length
"""
from numpy import ndarray
import scipy.constants
import numpy as np
from . import EngineerIO, returns_unit
from ..Units import UnknownUnitInContextException, Unit

__all__ = ["normalize_length", "convert_length_to_meters", "EngineerLengthIO"]

def _length_units():
    """
    Returns a set of length unit symbols
    """
    return {
        "m", "meter", "meters",
        "mm", "cm", "dm", "km", "µm", "um", "nm", "pm", "fm", "am",
        "mil", "in", "inch", "inches", '"',
        "ft", "foot", "feet", "yd", "yard",
        "mile", "miles", "nautical mile", "nautical miles",
        "pt", "point", "points",
        "AU", "au", "AUs", "ly", "lightyear", "lightyears", 
        "light year", "light years", "pc", "parsec", "parsecs",
        "Å", "Angstrom", "angstrom"
    }

def _length_unit_aliases():
    """
    Returns a dictionary mapping length unit aliases to their canonical symbols
    """
    return {
        # Metric aliases handled by SI prefixes
        # Imperial aliases
        "inches": "inch",
        "feet": "ft", 
        "yards": "yard",
        "miles": "mile",
        "points": "point",
        "mils": "mil",
        
        # Astronomical aliases
        "AUs": "AU",
        "lightyears": "ly",
        "light years": "ly", 
        "light year": "ly",
        "parsecs": "pc",
        
        # Other aliases
        "angstrom": "Å",
        "Angstrom": "Å",
        "nautical miles": "nautical mile"
    }

def _default_unit_prefix_map_length():
    """
    Returns the default unit prefix map for length calculations (includes c and d prefixes)
    """
    return {
        'y': -24, 'z': -21, 'a': -18, 'f': -15, 'p': -12,
        'n': -9, 'µ': -6, 'μ': -6, 'u': -6, 'm': -3,
        'c': -2, 'd': -1, '': 0,
        'k': 3, 'M': 6, 'G': 9, 'T': 12, 'P': 15,
        'E': 18, 'Z': 21, 'Y': 24
    }

_length_factors = {
    "": 1., # Assumed. It's SI!
    "m": 1.,
    "meter": 1.,
    "meters": 1.,
    'mil': 1e-3 * scipy.constants.inch,
    'in': scipy.constants.inch,
    'inch': scipy.constants.inch,
    'inches': scipy.constants.inch,
    '\"': scipy.constants.inch,
    'foot': scipy.constants.foot,
    'feet': scipy.constants.foot,
    'ft': scipy.constants.foot,
    'yd': scipy.constants.yard,
    'yard': scipy.constants.yard,
    'mile': scipy.constants.mile,
    'miles': scipy.constants.mile,
    'nautical mile': scipy.constants.nautical_mile,
    'nautical miles': scipy.constants.nautical_mile,
    'pt': scipy.constants.point,
    'point': scipy.constants.point,
    'points': scipy.constants.point,
    'AU': scipy.constants.astronomical_unit,
    'au': scipy.constants.astronomical_unit,
    'AUs': scipy.constants.astronomical_unit,
    'ly': scipy.constants.light_year,
    'lightyear': scipy.constants.light_year,
    'lightyears': scipy.constants.light_year,
    'light year': scipy.constants.light_year,
    'light years': scipy.constants.light_year,
    'pc': scipy.constants.parsec,
    'parsec': scipy.constants.parsec,
    'parsecs': scipy.constants.parsec,
    'Å': scipy.constants.angstrom,
    'Angstrom': scipy.constants.angstrom,
    'angstrom': scipy.constants.angstrom,
}


class EngineerLengthIO(EngineerIO):
    """
    EngineerIO subclass specialized for length unit parsing and conversion.
    """
    
    _instance = None
    
    def __init__(self):
        # Use length-specific configuration
        super().__init__(
            units=_length_units(),
            unit_aliases=_length_unit_aliases(),
            unit_prefix_map=_default_unit_prefix_map_length()
        )
    
    @classmethod
    def instance(cls):
        """
        Get the singleton instance of EngineerLengthIO
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @returns_unit("m")
    def normalize_length(self, s):
        """
        Normalize a length to meters.
        Returns the numeric value in m or None.

        NOTE: 1 nm is one nanometer, not one nautical mile! Use "1 nautical mile" instead!

        Valid inputs include:
        - "1.0" => 1.0
        - "1.0 mm" => 0.001
        - "1 inch" => 0.0254
        - "1 mil" => 0.000254
        - "1.2 M light years" => 1.135287656709696e+22
        - "9.15 kpc" => 2.8233949868947424e+17
        """
        if s is None:
            return None
        if isinstance(s, list):
            return [self.normalize_length(v) for v in s]
        if isinstance(s, ndarray):
            return np.asarray([self.normalize_length(v) for v in s])
        result = self.normalize(s)
        if result.unit in _length_factors:
            return result.value * _length_factors[result.unit]
        raise UnknownUnitInContextException(f"Unknown length unit: {result.unit}")
    
    @returns_unit("m")
    def convert_length_to_meters(self, value, unit):
        """
        Given a number or Engineer string (unit ignored) <value>
        in <unit>, convert it to meters.
        """
        # Currently a hack, but doing it directly will not parse SI units
        return self.normalize_length(f"{value} {unit}")


# Backward compatibility functions
@returns_unit("m")
def normalize_length(s, instance=None):
    """
    Normalize a length to meters.
    Returns the numeric value in m or None.

    NOTE: 1 nm is one nanometer, not one nautical mile! Use "1 nautical mile" instead!

    Valid inputs include:
    - "1.0" => 1.0
    - "1.0 mm" => 0.001
    - "1 inch" => 0.0254
    - "1 mil" => 0.000254
    - "1.2 M light years" => 1.135287656709696e+22
    - "9.15 kpc" => 2.8233949868947424e+17
    """
    if instance is None:
        instance = EngineerLengthIO.instance()
    return instance.normalize_length(s)

@returns_unit("m")
def convert_length_to_meters(value, unit, instance=None):
    """
    Given a number or Engineer string (unit ignored) <value>
    in <unit>, convert it to meters.
    """
    if instance is None:
        instance = EngineerLengthIO.instance()
    return instance.convert_length_to_meters(value, unit)

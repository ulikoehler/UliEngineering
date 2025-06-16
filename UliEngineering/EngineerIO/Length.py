#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for length
"""
from numpy import ndarray
import scipy.constants
import numpy as np

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Defaults import default_si_prefix_map
from . import EngineerIO
from .UnitInfo import EngineerIOConfiguration, UnitInfo
from ..Units import UnknownUnitInContextException

__all__ = ["normalize_length", "convert_length_to_meters", "EngineerLengthIO"]

def _length_unit_infos():
    """
    Returns a list of UnitInfo objects for length units
    """
    return [
        # Base meter unit
        UnitInfo('m', 1.0, ['meter', 'meters']),
        
        # Imperial units
        UnitInfo('mil', 1e-3 * scipy.constants.inch, ['mils']),
        UnitInfo('in', scipy.constants.inch, ['"', 'inch', 'inches']),
        UnitInfo('ft', scipy.constants.foot, ['foot', 'feet']),
        UnitInfo('yd', scipy.constants.yard, ['yard', 'yards']),
        UnitInfo('mile', scipy.constants.mile, ['miles']),
        UnitInfo('nautical mile', scipy.constants.nautical_mile, ['nautical miles']),
        UnitInfo('pt', scipy.constants.point, ['point', 'points']),
        
        # Astronomical units
        UnitInfo('AU', scipy.constants.astronomical_unit, ['au', 'AUs']),
        UnitInfo('ly', scipy.constants.light_year, ['lightyear', 'lightyears', 'light years', 'light year']),
        UnitInfo('pc', scipy.constants.parsec, ['parsec', 'parsecs']),
        
        # Other units
        UnitInfo('Å', scipy.constants.angstrom, ['angstrom', 'Angstrom']),
    ]

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

def _create_length_config():
    """
    Create a custom EngineerIOConfiguration for length units with extended SI prefixes
    """
    config = EngineerIOConfiguration.default()
    return EngineerIOConfiguration(
        units=_length_unit_infos(),
        unit_prefixes=config.unit_prefixes,
        si_prefix_map=default_si_prefix_map(include_length_unit_prefixes=True)
    )

class EngineerLengthIO(EngineerIO):
    """
    EngineerIO subclass specialized for length unit parsing and conversion.
    """
    _instance = None
    
    def __init__(self):
        # Use length-specific configuration
        super().__init__(config=_create_length_config())
    
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
        return self.normalize_numeric(s)
    
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

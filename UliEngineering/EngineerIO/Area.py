#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for area
"""
from numpy import ndarray
import scipy.constants
import numpy as np

from UliEngineering.EngineerIO.Types import NormalizeResult
from . import EngineerIO
from .Decorators import returns_unit
from .UnitInfo import EngineerIOConfiguration, UnitAlias, UnitInfo
from .Defaults import default_si_prefix_map

__all__ = ["normalize_area", "convert_area_to_square_meters", "EngineerAreaIO"]

def area_unit_infos():
    return [
        # Base square meter unit
        UnitInfo('m²', 1.0, ['m^2', 'square meter', 'square meters', 'sq m', 'sqm']),
        
        # SI-prefixed meter units (these will be handled by SI prefix parsing)
        UnitAlias('mm²', aliases=['mm^2', 'square millimeter', 'square millimeters', 'square mm', 'sq mm', 'mm sq', 'mm squared', 'millimeter squared']),
        UnitAlias('cm²', aliases=['cm^2', 'square centimeter', 'square centimeters', 'square cm', 'sq cm', 'cm sq', 'cm squared', 'centimeter squared', 'centimeters squared']),
        UnitAlias('dm²', aliases=['dm^2', 'square decimeter', 'square decimeters', 'square dm', 'sq dm', 'dm sq', 'dm squared', 'decimeter squared', 'decimeters squared']),
        UnitAlias('km²', aliases=['km^2', 'square kilometer', 'square kilometers', 'square km', 'sq km', 'km squared', 'km sq', 'kilometers sq', 'kilometers squared']),
        UnitAlias('µm²', aliases=['µm^2', 'um²', 'um^2', 'square micrometer', 'square micrometers', 'square µm', 'sq µm', 'sq um', 'um sq', 'µm sq', 'µm squared', 'micrometer squared', 'micrometers squared']),
        UnitAlias('nm²', aliases=['nm^2', 'square nanometer', 'square nanometers', 'square nm', 'sq nm', 'nm sq', 'nm squared', 'nanometer squared', 'nanometers squared']),
        
        # Imperial area units
        UnitInfo('in²', scipy.constants.inch**2, ['in^2', 'square inch', 'square inches', 'sq in']),
        UnitInfo('ft²', scipy.constants.foot**2, ['ft^2', 'square foot', 'square feet', 'sq ft']),
        UnitInfo('yd²', scipy.constants.yard**2, ['yd^2', 'square yard', 'square yards', 'sq yd']),
        
        # Agricultural units
        UnitInfo('acre', 4046.8564224, ['acres']),
        UnitInfo('ha', 10000.0, ['hectare', 'hectares']),
        UnitInfo('are', 100.0, ['ares']),
        
        # Scientific units
        UnitInfo('barn', 1e-28, ['barns', 'b']),
        
        # Atomic and molecular cross-section units
        UnitInfo('Å²', 1e-20, ['angstrom squared', 'angstrom^2', 'A^2', 'A²']),
        UnitInfo('bohr²', (scipy.constants.physical_constants['Bohr radius'][0])**2, 
                ['bohr squared', 'bohr^2', 'a0²', 'a0^2', 'atomic unit of area']),
        
        # Astronomical units
        UnitInfo('AU²', (scipy.constants.au)**2, ['AU^2', 'astronomical unit squared']),
        UnitInfo('pc²', (scipy.constants.parsec)**2, ['parsec squared', 'parsec^2']),
        UnitInfo('ly²', (scipy.constants.c * scipy.constants.Julian_year)**2, 
                ['light year squared', 'light-year squared', 'ly^2', 'lightyear²']),
        
        # Additional metric units
        UnitInfo('ca', 1.0, ['centiare', 'centiares']),  # Same as m²
        UnitInfo('decare', 1000.0, ['decares']),  # 10 ares
        
        # Planck area
        UnitInfo('lP²', (scipy.constants.physical_constants['Planck length'][0])**2, 
                ['Planck area', 'planck area', 'lp^2', 'lp²']),
    ]

def _create_area_config():
    """
    Create a custom EngineerIOConfiguration for area units with extended SI prefixes
    """
    config = EngineerIOConfiguration.default()
    return EngineerIOConfiguration(
        units=area_unit_infos(),
        unit_prefixes=config.unit_prefixes,
        si_prefix_map=default_si_prefix_map(include_length_unit_prefixes=True)
    )

class EngineerAreaIO(EngineerIO):
    """
    EngineerIO subclass specialized for area unit parsing and conversion.
    """
    
    _instance = None
    
    def __init__(self):
        # Use area-specific configuration
        super().__init__(config=_create_area_config())
    
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
        print(s, self.normalize(s), self._apply_squaredness_to_value(self.normalize(s)))
        # We can't just normalize() it here, need to handle squaredness
        return self._apply_squaredness_to_value(self.normalize(s))
            
    def _apply_squaredness_to_value(self, result: NormalizeResult):
        # We need to takes care of the fact that cm² means (cm)², nm² means (nm)²
        # whereas µbarn means micro(barns), hence the multiplier does not need to be squared.
        # Heuristic: everything with square in it (², ^2 etc) must be squared
        is_square_unit = False
        # NOTE: All squared units must be aliased to a ² unit
        if "²" in result.unit:
            # We have a square unit, so we need to square the value
            is_square_unit = True
        # NOTE: The value has already been multiplied by the prefix factor ONCE,
        # so we need to multiply it again (to make it squared in total)
        if is_square_unit:
            return result.value * result.prefix_multiplier
        else: # No need to modify
            return result.value
    
    @returns_unit("m²")
    def convert_area_to_square_meters(self, value, unit):
        """
        Given a number or Engineer string (unit ignored) <value>
        in <unit>, convert it to square meters.
        """
        # Currently a hack, but doing it directly will not parse SI units
        return self.normalize_area(f"{value} {unit}")


@returns_unit("m²")
def convert_area_to_square_meters(value, unit, instance=None):
    """
    Given a number or Engineer string (unit ignored) <value>
    in <unit>, convert it to square meters.
    """
    if instance is None:
        instance = EngineerAreaIO.instance()
    return instance.convert_area_to_square_meters(value, unit)

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
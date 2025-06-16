#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for volume
"""
from numpy import ndarray
import scipy.constants
import numpy as np

from UliEngineering.EngineerIO.Types import NormalizeResult
from . import EngineerIO
from .Decorators import returns_unit
from .UnitInfo import EngineerIOConfiguration, UnitAlias, UnitInfo
from .Defaults import default_si_prefix_map

__all__ = ["normalize_volume", "convert_volume_to_cubic_meters", "EngineerVolumeIO"]

def volume_unit_infos():
    return [
        # Base cubic meter unit
        UnitInfo('m³', 1.0, ['m^3', 'cubic meter', 'cubic meters', 'cu m', 'cbm']),
        
        # SI-prefixed meter units (these will be handled by SI prefix parsing)
        UnitAlias('mm³', aliases=['mm^3', 'cubic millimeter', 'cubic millimeters', 'cubic mm', 'cu mm', 'mm cu', 'mm cubed', 'millimeter cubed']),
        UnitAlias('cm³', aliases=['cm^3', 'cubic centimeter', 'cubic centimeters', 'cubic cm', 'cu cm', 'cm cu', 'cm cubed', 'centimeter cubed', 'centimeters cubed', 'cc']),
        UnitAlias('dm³', aliases=['dm^3', 'cubic decimeter', 'cubic decimeters', 'cubic dm', 'cu dm', 'dm cu', 'dm cubed', 'decimeter cubed', 'decimeters cubed']),
        UnitAlias('km³', aliases=['km^3', 'cubic kilometer', 'cubic kilometers', 'cubic km', 'cu km', 'km cubed', 'km cu', 'kilometers cu', 'kilometers cubed']),
        UnitAlias('µm³', aliases=['µm^3', 'um³', 'um^3', 'cubic micrometer', 'cubic micrometers', 'cubic µm', 'cu µm', 'cu um', 'um cu', 'µm cu', 'µm cubed', 'micrometer cubed', 'micrometers cubed']),
        UnitAlias('nm³', aliases=['nm^3', 'cubic nanometer', 'cubic nanometers', 'cubic nm', 'cu nm', 'nm cu', 'nm cubed', 'nanometer cubed', 'nanometers cubed']),
        
        # Imperial volume units
        UnitInfo('in³', scipy.constants.inch**3, ['in^3', 'cubic inch', 'cubic inches', 'cu in']),
        UnitInfo('ft³', scipy.constants.foot**3, ['ft^3', 'cubic foot', 'cubic feet', 'cu ft']),
        UnitInfo('yd³', scipy.constants.yard**3, ['yd^3', 'cubic yard', 'cubic yards', 'cu yd']),
        
        # Liquid volume units (US liquid measurements)
        UnitInfo('L', 0.001, ['liter', 'liters', 'litre', 'litres']),
        UnitInfo('gal', 0.003785411784, ['gallon', 'gallons']),  # US liquid gallon
        UnitInfo('qt', 0.000946352946, ['quart', 'quarts']),    # US liquid quart
        UnitInfo('pt', 0.000473176473, ['pint', 'pints']),      # US liquid pint
        UnitInfo('fl oz', 2.95735296875e-05, ['fluid ounce', 'fluid ounces', 'floz']),  # US fluid ounce
        UnitInfo('cup', 0.0002365882365, ['cups']),             # US cup
        UnitInfo('tbsp', 1.47867648437e-05, ['tablespoon', 'tablespoons']),  # US tablespoon
        UnitInfo('tsp', 4.92892161458e-06, ['teaspoon', 'teaspoons']),       # US teaspoon
        
        # Oil and gas industry units
        UnitInfo('bbl', 0.158987294928, ['barrel', 'barrels']), # US oil barrel
        
        # Scientific units
        UnitInfo('Å³', 1e-30, ['angstrom cubed', 'angstrom^3', 'A^3', 'A³']),
        UnitInfo('bohr³', (scipy.constants.physical_constants['Bohr radius'][0])**3, 
                ['bohr cubed', 'bohr^3', 'a0³', 'a0^3', 'atomic unit of volume']),
        
        # Astronomical units
        UnitInfo('AU³', (scipy.constants.au)**3, ['AU^3', 'astronomical unit cubed']),
        UnitInfo('pc³', (scipy.constants.parsec)**3, ['parsec cubed', 'parsec^3']),
        UnitInfo('ly³', (scipy.constants.c * scipy.constants.Julian_year)**3, 
                ['light year cubed', 'light-year cubed', 'ly^3', 'lightyear³']),
        
        # Planck volume
        UnitInfo('lP³', (scipy.constants.physical_constants['Planck length'][0])**3, 
                ['Planck volume', 'planck volume', 'lp^3', 'lp³']),
    ]

def _create_volume_config():
    """
    Create a custom EngineerIOConfiguration for volume units with extended SI prefixes
    """
    config = EngineerIOConfiguration.default()
    return EngineerIOConfiguration(
        units=volume_unit_infos(),
        unit_prefixes=config.unit_prefixes,
        si_prefix_map=default_si_prefix_map(include_length_unit_prefixes=True)
    )

class EngineerVolumeIO(EngineerIO):
    """
    EngineerIO subclass specialized for volume unit parsing and conversion.
    """
    
    _instance = None
    
    def __init__(self):
        # Use volume-specific configuration
        super().__init__(config=_create_volume_config())
    
    @classmethod
    def instance(cls):
        """
        Get the singleton instance of EngineerVolumeIO
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @returns_unit("m³")
    def normalize_volume(self, s):
        """
        Normalize a volume to cubic meters.
        Returns the numeric value in m³, a list or ndarray of converted values,
        or None if the input is None.

        Valid inputs include:
        - "1.0" => 1.0
        - "1.0 cm³" => 0.000001
        - "1 cubic inch" => 1.6387064e-05
        - "1 liter" => 0.001
        - "1 gallon" => 0.003785411784
        - "100 cu ft" => 2.8316846592
        """
        if s is None:
            return None
        if isinstance(s, list):
            return [self.normalize_volume(v) for v in s]
        if isinstance(s, ndarray):
            return np.asarray([self.normalize_volume(v) for v in s])
        # We can't just normalize() it here, need to handle cubedness
        return self._apply_cubedness_to_value(self.normalize(s))
            
    def _apply_cubedness_to_value(self, result: NormalizeResult):
        # We need to takes care of the fact that cm³ means (cm)³, nm³ means (nm)³
        # whereas µL means micro(liters), hence the multiplier does not need to be cubed.
        # Heuristic: everything with cubic in it (³, ^3 etc) must be cubed
        is_cubic_unit = False
        # NOTE: All cubed units must be aliased to a ³ unit
        if "³" in result.unit:
            # We have a cubic unit, so we need to cube the value
            is_cubic_unit = True
        # NOTE: The value has already been multiplied by the prefix factor ONCE,
        # so we need to multiply it again twice more (to make it cubed in total)
        if is_cubic_unit:
            return result.value * result.prefix_multiplier * result.prefix_multiplier
        else: # No need to modify
            return result.value
    
    @returns_unit("m³")
    def convert_volume_to_cubic_meters(self, value, unit):
        """
        Given a number or Engineer string (unit ignored) <value>
        in <unit>, convert it to cubic meters.
        """
        # Currently a hack, but doing it directly will not parse SI units
        return self.normalize_volume(f"{value} {unit}")


@returns_unit("m³")
def convert_volume_to_cubic_meters(value, unit, instance=None):
    """
    Given a number or Engineer string (unit ignored) <value>
    in <unit>, convert it to cubic meters.
    """
    if instance is None:
        instance = EngineerVolumeIO.instance()
    return instance.convert_volume_to_cubic_meters(value, unit)

@returns_unit("m³")
def normalize_volume(s, instance=None):
    """
    Normalize a volume to cubic meters.
    Returns the numeric value in m³, a list or ndarray of converted values,
    or None if the input is None.

    Valid inputs include:
    - "1.0" => 1.0
    - "1.0 cm³" => 0.000001
    - "1 cubic inch" => 1.6387064e-05
    - "1 liter" => 0.001
    - "1 gallon" => 0.003785411784
    - "100 cu ft" => 2.8316846592
    """
    if instance is None:
        instance = EngineerVolumeIO.instance()
    return instance.normalize_volume(s)

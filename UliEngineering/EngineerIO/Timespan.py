#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timespan normalization and conversion utilities for UliEngineering
"""
import numpy as np

from UliEngineering.EngineerIO.Decorators import returns_unit
from . import EngineerIO
from .UnitInfo import UnitInfo, UnitAlias

def _timespan_unit_infos():
    """
    Returns a list of UnitInfo and UnitAlias objects for timespan units
    """
    return [
        # SI-prefixed seconds (using UnitAlias)
        UnitAlias('as', aliases=['s', 'attosecond', 'attoseconds', 'asec', 'asecs']),
        UnitAlias('fs', aliases=['s', 'femtosecond', 'femtoseconds', 'fsec', 'fsecs']),
        UnitAlias('ps', aliases=['s', 'picosecond', 'picoseconds', 'psec', 'psecs']),
        UnitAlias('ns', aliases=['s', 'nanosecond', 'nanoseconds', 'nsec', 'nsecs']),
        UnitAlias('µs', aliases=['s', 'microsecond', 'microseconds', 'µsecond', 'µsec', 'usec', 'us']),
        UnitAlias('ms', aliases=['s', 'millisecond', 'milliseconds']),
        
        # Base seconds unit
        UnitInfo('s', aliases=['second', 'seconds', 'sec', 'secs']),
        
        # Units with custom conversion factors
        UnitInfo('min', 60, ['minute', 'minutes']),
        UnitInfo('h', 3600, ['hour', 'hours']),
        UnitInfo('d', 86400, ['day', 'days']),
        UnitInfo('w', 604800, ['week', 'weeks']),
        UnitInfo('mo', 2629746, ['month', 'months']),  # 1/12th of a year
        UnitInfo('y', 31556952, ['year', 'years']),  # 365.2425 days on average
        UnitInfo('decade', 315569520, ['decades']),
        UnitInfo('century', 3155695200, ['centuries']),
        UnitInfo('millenium', 31556952000, ['millenia', 'millennium', 'millennia']),
        
        # SI-prefixed years (using UnitAlias)
        UnitAlias('My', aliases=['megayear', 'megayears', 'Myr', 'Myrs']),
        UnitAlias('Gy', aliases=['gigayear', 'gigayears', 'Gyr', 'Gyrs']),
        UnitAlias('Ty', aliases=['terayear', 'terayears', 'Tyr', 'Tyrs']),
    ]

def _timespan_factors():
    """
    Returns a dictionary mapping timespan units to their conversion factors to seconds
    """
    factors = {}
    for item in _timespan_unit_infos():
        if isinstance(item, UnitInfo):
            factors[item.canonical] = item.factor
        # UnitAlias objects don't contribute to factors - they reference existing units
    return factors

def _timespan_unit_aliases():
    """
    Maps verbose timespan unit names to their compact symbols.
    """
    aliases = {}
    for item in _timespan_unit_infos():
        if isinstance(item, UnitInfo):
            for alias in item.aliases:
                aliases[alias] = item.canonical
        elif isinstance(item, UnitAlias):
            for alias in item.aliases:
                aliases[alias] = item.canonical
    return aliases

def _timespan_units():
    """
    Returns a set of timespan unit symbols
    """
    units = set()
    for item in _timespan_unit_infos():
        if isinstance(item, UnitInfo):
            units.add(item.canonical)
    return units

class EngineerTimespanIO(EngineerIO):
    """
    Specialized EngineerIO class for timespan operations
    """
    def __init__(self):
        # Initialize with timespan unit infos
        super().__init__(
            unit_infos=_timespan_unit_infos(),
            timespan_units=_timespan_factors()
        )
    
    @returns_unit("s")
    def normalize_timespan(self, arg: str | bytes | int | float | np.generic | np.ndarray) -> int | float | np.generic | np.ndarray:
        """
        Normalize a given timespan to SI units (seconds).
        Numeric inputs are assumed to be in seconds.
        """
        return self.normalize_numeric(arg)

    @classmethod
    def instance(cls):
        """
        Get the singleton instance of EngineerTimespanIO
        """
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = cls()
        return cls._instance

def normalize_timespan(v: str | bytes | int | float | np.generic | np.ndarray) -> int | float | np.generic | np.ndarray:
    """
    Normalize a given timespan to SI units (seconds).
    Numeric inputs are assumed to be in seconds.
    """
    return EngineerTimespanIO.instance().normalize_timespan(v)
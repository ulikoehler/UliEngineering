#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timespan normalization and conversion utilities for UliEngineering
"""
import numpy as np

from UliEngineering.EngineerIO.Decorators import returns_unit
from . import EngineerIO
from .UnitInfo import UnitInfo, UnitAlias, EngineerIOConfiguration

def _timespan_unit_infos():
    """
    Returns a list of UnitInfo and UnitAlias objects for timespan units
    """
    return [
        # SI-prefixed seconds (using UnitAlias)
        UnitAlias('as', aliases=['attosecond', 'attoseconds', 'asec', 'asecs']),
        UnitAlias('fs', aliases=['femtosecond', 'femtoseconds', 'fsec', 'fsecs']),
        UnitAlias('ps', aliases=['picosecond', 'picoseconds', 'psec', 'psecs']),
        UnitAlias('ns', aliases=['nanosecond', 'nanoseconds', 'nsec', 'nsecs']),
        UnitAlias('µs', aliases=['microsecond', 'microseconds', 'µsecond', 'µsec', 'usec', 'us']),
        UnitAlias('ms', aliases=['millisecond', 'milliseconds']),
        
        # Base seconds unit
        UnitInfo('s', 1.0, ['second', 'seconds', 'sec', 'secs']),
        
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

def _create_timespan_config():
    """
    Create a custom EngineerIOConfiguration for timespan units
    """
    config = EngineerIOConfiguration.default()
    return EngineerIOConfiguration(
        units=_timespan_unit_infos(),
        unit_prefixes=config.unit_prefixes,
        si_prefix_map=config.si_prefix_map
    )

class EngineerTimespanIO(EngineerIO):
    """
    Specialized EngineerIO class for timespan operations
    """
    
    _instance = None
    
    def __init__(self):
        # Use timespan-specific configuration
        super().__init__(config=_create_timespan_config())
    
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
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def normalize_timespan(v: str | bytes | int | float | np.generic | np.ndarray) -> int | float | np.generic | np.ndarray:
    """
    Normalize a given timespan to SI units (seconds).
    Numeric inputs are assumed to be in seconds.
    """
    return EngineerTimespanIO.instance().normalize_timespan(v)
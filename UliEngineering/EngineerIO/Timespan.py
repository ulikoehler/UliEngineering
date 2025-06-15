#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timespan normalization and conversion utilities for UliEngineering
"""
import numpy as np
from . import returns_unit, EngineerIO
from ..Utils.String import partition_at_numeric_to_nonnumeric_boundary

def _timespan_factors():
    """
    Returns a dictionary mapping timespan units to their conversion factors to seconds
    """
    return {
        # Attoseconds
        'as': 1e-18,
        # Femtoseconds
        'fs': 1e-15,
        # Picoseconds
        'ps': 1e-12,
        # Nanoseconds
        'ns': 1e-9,
        # Microseconds
        'µs': 1e-6,
        'us': 1e-6,
        # Milliseconds
        'ms': 0.001,
        # seconds
        's': 1,
        # Minutes
        'min': 60,
        # hours
        'h': 3600,
        # days
        'd': 86400,
        # weeks
        'w': 604800,
        # Months (we're using the average duration of a month)
        'mo': 2629746,  # 1/12th of a year, see below for the definition of a year
        # years (365.2425 days on average)
        'y': 31556952,
        # Decades
        'decade': 315569520,
        # Centuries
        'century': 3155695200,
        # Millenia
        'millenium': 31556952000,
    }

def _timespan_unit_aliases():
    """
    Maps verbose timespan unit names to their compact symbols.
    """
    return {
        # Attosecond aliases
        'attosecond': 'as',
        'attoseconds': 'as',
        'asec': 'as',
        'asecs': 'as',
        
        # Femtosecond aliases
        'femtosecond': 'fs',
        'femtoseconds': 'fs',
        'fsec': 'fs',
        'fsecs': 'fs',
        
        # Picosecond aliases
        'picosecond': 'ps',
        'picoseconds': 'ps',
        'psec': 'ps',
        'psecs': 'ps',
        
        # Nanosecond aliases
        'nanosecond': 'ns',
        'nanoseconds': 'ns',
        'nsec': 'ns',
        'nsecs': 'ns',
        
        # Microsecond aliases
        'microsecond': 'µs',
        'microseconds': 'µs',
        'µsecond': 'µs',
        'µsec': 'µs',
        'usec': 'µs',
        
        # Millisecond aliases
        'millisecond': 'ms',
        'milliseconds': 'ms',
        
        # Second aliases
        'second': 's',
        'seconds': 's',
        'sec': 's',
        'secs': 's',
        
        # Minute aliases
        'minute': 'min',
        'minutes': 'min',
        
        # Hour aliases
        'hour': 'h',
        'hours': 'h',
        
        # Day aliases
        'day': 'd',
        'days': 'd',
        
        # Week aliases
        'week': 'w',
        'weeks': 'w',
        
        # Month aliases
        'month': 'mo',
        'months': 'mo',
        
        # Year aliases
        'year': 'y',
        'years': 'y',
        
        # Decade aliases
        'decade': 'decade',
        'decades': 'decade',
        
        # Century aliases
        'century': 'century',
        'centuries': 'century',
        
        # Millennium aliases
        'millenium': 'millenium',
        'millenia': 'millenium',
        'millennium': 'millenium',
        'millennia': 'millenium',
        
        # Megayear aliases
        'megayear': 'My',
        'megayears': 'My',
        'Myr': 'My',
        'Myrs': 'My',
        
        # Gigayear aliases
        'gigayear': 'Gy',
        'gigayears': 'Gy',
        'Gyr': 'Gy',
        'Gyrs': 'Gy',
        
        # Terayear aliases
        'terayear': 'Ty',
        'terayears': 'Ty',
        'Tyr': 'Ty',
        'Tyrs': 'Ty',
    }

def _timespan_units():
    """
    Returns a set of timespan unit symbols
    """
    return set(_timespan_factors().keys())

class EngineerTimespanIO(EngineerIO):
    """
    Specialized EngineerIO class for timespan operations
    """
    
    def __init__(self):
        # Initialize with standard units plus timespan units
        super().__init__(
            units=_timespan_units(),
            unit_aliases=_timespan_unit_aliases(),
            timespan_units=_timespan_factors()
        )
    
    @returns_unit("s")
    def normalize_timespan(self, arg: str | bytes | int | float | np.generic | np.ndarray) -> int | float | np.generic | np.ndarray:
        """
        Normalize a given timespan to SI units (seconds).
        Numeric inputs are assumed to be in seconds.
        """
        if isinstance(arg, bytes):
            arg = arg.decode("utf8")
        if isinstance(arg, (int, float)):
            return arg # Already a number. Just return!
        elif isinstance(arg, (str)):
            s, unit = partition_at_numeric_to_nonnumeric_boundary(arg) # Remove unit
            s, unit = s.strip(), unit.strip()
            if not s:
                raise ValueError(f"Empty value in timespan: {arg}")
            if not unit: # Assume seconds (SI unit of time)
                return float(s)
            # Resolve unit alias if it exists
            resolved_unit = self.unit_aliases.get(unit, unit)
            # Check if unit exists in timespan_units
            if resolved_unit not in self.timespan_units:
                raise ValueError(f"Invalid timespan unit '{unit}' in '{arg}'. Expected one of {list(self.timespan_units.keys())}")
            return float(s) * self.timespan_units[resolved_unit]
        elif isinstance(arg, (np.ndarray, list)):
            return np.vectorize(self.normalize_timespan)(arg)
        else:
            raise ValueError(f"Unsupported type for normalization: {type(arg)}")

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
#!/usr/bin/env python3
"""Utilities for handling value ranges."""
from collections import namedtuple
from UliEngineering.EngineerIO import normalize_numeric, format_value

__all__ = ["ValueRange", "normalize_minmax_tuple"]

_ValueRange = namedtuple("ValueRange", ["min", "max", "unit"])

class ValueRange(_ValueRange):
    
    """Represent a range of values with optional unit."""

    def __new__(cls, min_val, max_val, unit=None, significant_digits=4):
        """Create a new ValueRange instance."""
        self = super(ValueRange, cls).__new__(cls, min_val, max_val, unit)
        self.significant_digits = significant_digits
        return self

    def __repr__(self):
        """Return string representation of the ValueRange."""
        return f"ValueRange('{format_value(self.min, self.unit, significant_digits=self.significant_digits)}', '{format_value(self.max, self.unit, significant_digits=self.significant_digits)}')"

    @property
    def minmax(self):
        """Return (min, max). Utility e.g. for unpacking a ValueRange ignoring Unit.

        Returns
        -------
        tuple
            A tuple containing the minimum and maximum values of the range.
        
        """
        return (self.min, self.max)

def normalize_minmax_tuple(arg, name="field"):
    """Interpret arg either as a single +- value or as a 2-tuple of + and - values.

    If arg is a tuple:
        Return ValueRange(arg[0], arg[1]) (strings are normalized)
    Else:
        Return ValueRange(-arg, +arg) (strings are normalized)

    Parameters
    ----------
    arg : float or tuple
        The input value(s) to normalize.
    name : str, optional
        The name of the field being normalized. Defaults to "field".

    Returns
    -------
    ValueRange
        A normalized ValueRange instance.
    
    """
    # Parse coefficient and compute min & max factors
    if isinstance(arg, tuple):
        # Check length 2
        if len(arg) != 2:
            raise ValueError(f"If {name} is given as a tuple, it must have length 2. {name} is {arg}")
        # Parse tuple
        min_value = normalize_numeric(arg[0])
        max_value = normalize_numeric(arg[1])
    else:
        arg = normalize_numeric(arg)
        min_value = -arg
        max_value = arg
    return ValueRange(min_value, max_value)

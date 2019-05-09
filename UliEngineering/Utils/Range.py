#!/usr/bin/env python3
from collections import namedtuple
from UliEngineering.EngineerIO import normalize_numeric, format_value

__all__ = ["ValueRange", "normalize_minmax_tuple"]

_ValueRange = namedtuple("ValueRange", ["min", "max", "unit"])

class ValueRange(_ValueRange):
    def __new__(cls, min, max, unit=None, significant_digits=4):
        self = super(ValueRange, cls).__new__(cls, min, max, unit)
        self.significant_digits = significant_digits
        return self

    def __repr__(self):
        return "ValueRange('{}', '{}')".format(
            format_value(self.min, self.unit, significant_digits=self.significant_digits),
            format_value(self.max, self.unit, significant_digits=self.significant_digits)
        )

def normalize_minmax_tuple(arg, name="field"):
    """
    Interprets arg either a single +- value or as
    a 2-tuple of + and - values.
    All vaues 

    If arg is a tuple:
        Return ValueRange(arg[0], arg[1]) (strings are normalized)
    Else:
        Return ValueRange(-arg, +arg) (strings are normalized)

    name is for debugging purposes and shown in the exception string
    """
    # Parse coefficient and compute min & max factors
    if isinstance(arg, tuple):
        # Check length 2
        if len(arg) != 2:
            raise ValueError("If {} is given as a tuple, it must have length 2. {} is {}".format(name, name, arg))
        # Parse tuple
        min_value = normalize_numeric(arg[0])
        max_value = normalize_numeric(arg[1])
    else:
        arg = normalize_numeric(arg)
        min_value = -arg
        max_value = arg
    return ValueRange(min_value, max_value)

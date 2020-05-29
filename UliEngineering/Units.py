#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Units and exceptions related to units.
"""
from collections import namedtuple
import functools
import re

__all__ = ["Unit", "SubUnit", "UnannotatedReturnValueError",
           "InvalidUnitInContextException", "InvalidUnitCombinationException",
           "find_returned_unit", "UnknownUnitInContextException"]

_multiplicative_separator_re = re.compile(r"[\*·×]")

class SubUnit(object):
    """
    Represents a single (non-compound) unit to the power of N.
    For example, this class can represent:
        - m
        - m²
        - km³
        - N
        - Pa
        - m
    but not:
        - N/mm²
        - kg*N
    """
    def __init__(self, unit, power=1):
        self.unit = unit
        self.power = power

    @staticmethod
    def parse(s):
        """
        Parse a SubUnit from a string.
        """
        # Normalize ! 
        s = s.replace("²", "^2") \
             .replace("³", "^3") \
             .replace("⁴", "^4") \
             .replace("⁵", "^5") \
             .replace("⁶", "^6") \
             .replace("⁷", "^7") \
             .replace("⁸", "^8") \
             .replace("⁹", "^9") \
             .replace("⁰", "^0") \
             .replace("¹", "^1") \
        # Parse power
        power_count = s.count("^")
        if power_count == 0:
            # Only numerator
            unit = s
            power = 1
        elif power_count == 1:
            unit_str, _, power_str = s.partition("^")
            unit = unit_str
            power = int(power_str)
        else:
            raise ValueError(f"SubUnit '{s}' contains more than one slash. Can't process !")
        return SubUnit(unit, power)

    def __eq__(self, other):
        if not isinstance(other, SubUnit):
            raise NotImplementedError
        return self.unit == other.unit and self.power == other.power

    def __repr__(self):
        if self.power == 0:
            return ""
        elif self.power == 1:
            return self.unit
        else:
            # TODO pretty-format
            return f"{self.unit}^{self.power}"

class Unit(object):
    """
    Represents a potentially compound unit.
    For example, this class can represent:

    For example, this class can represent:
        - m
        - m²
        - km³
        - N
        - Pa
        - m
        - N/mm²
        - N*m
    """
    def __init__(self, unit):
        # Split unit into numerator & denominator
        slash_count = unit.count("/")
        if slash_count == 0:
            # Only numerator
            self.numerator = Unit._split_unit_string(unit)
            self.denominator = []
        elif slash_count == 1:
            num_str, _, den_str = unit.partition("/")
            self.numerator = Unit._split_unit_string(num_str)
            self.denominator = Unit._split_unit_string(den_str)
        else:
            raise ValueError(f"Unit '{unit}' contains more than one slash. Can't process !")
        # Sort for reproducible results
        self.numerator.sort()
        self.denominator.sort()

    @staticmethod
    def _split_unit_string(s):
        """
        Split a multiplied list of units into individual unit-power combinations
        """
        return [SubUnit.parse(entry) for entry in _multiplicative_separator_re.split(s)]

    def __repr__(self):
        numerator_str = "·".join(str(num) for num in self.numerator) if self.numerator else "1"
        if self.denominator:
            denominator_str = "·".join(str(den) for den in self.denominator)
            return f"{numerator_str}/{denominator_str}"
        else:
            # No denominator
            return numerator_str

    def __mul__(self, other):
        """
        Multiply this unit with another Unit.
        """
        if not isinstance(other, Unit):
            raise ValueError(f"Can't multiply a Unit with a non-Unit like {other}")
        pass # TODO

    def __eq__(self, other):
        if isinstance(other, (str, bytes)):
            return self == Unit(other)
        if not isinstance(other, Unit):
            raise NotImplementedError
        return str(self) == str(other)

    def __hash__(self):
        return self.__repr__().__hash__()

    def __bool__(self):
        """
        Units are true-ish if they are not dimensionless
        """
        return bool(self.numerator) or bool(self.denominator)



class UnannotatedReturnValueError(Exception):
    """
    Raised if the automatic unit finder cannot find
    the appropriate function annotation that
    tells an auto-formatting function which unit is being used.

    Returns the unit string
    """
    pass


class InvalidUnitInContextException(ValueError):
    """
    Raised if the unit might not be a globally
    unknown or invalid unit, but in the given context
    it can't be used
    """
    pass


class UnknownUnitInContextException(ValueError):
    """
    Raised if the unit is not known in this context,
    e.g. if "A" is used as a unit of length.

    The message should contain information on what type of
    quantity (e.g. length) is accepted.
    """
    pass

class InvalidUnitCombinationException(ValueError):
    """
    Raised if the units involved in an operation can't be
    combined in the way requested, for example if the
    """
    pass

def find_returned_unit(fn):
    """
    Given a function that is assumed to return a quantity
    and annotated with the corresponding unit, determines
    which is the unit returned by the function
    """
    if not callable(fn):
        raise ValueError("fn must be callable")
    # Access innermost function inside possibly nested partials
    annotatedFN = fn
    while isinstance(annotatedFN, functools.partial):
        annotatedFN = annotatedFN.func
    # We have the innermost function
    try:
        unit = annotatedFN.__annotations__["return"]
        # Assume it's a Unit namedtuple
        return unit
    except KeyError: # No return annotation
        raise UnannotatedReturnValueError(
            "Function {} does not have an annotated return value".format(fn))

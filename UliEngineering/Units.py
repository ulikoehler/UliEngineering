#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Units and exceptions related to units.
"""
from collections import namedtuple
import functools

__all__ = ["Unit", "UnannotatedReturnValueError",
           "InvalidUnitInContextException", "InvalidUnitCombinationException",
           "find_returned_unit", "UnknownUnitInContextException"]

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
    def __init__(self, unit, power=None):
        # Only one argument?
        if power is None:
            self.unit, self.power = SubUnit.parse(unit)

    @staticmethod
    def parse(self, s):
        """
        Parse a SubUnit from a string.
        """
        # Normalize ! 
        s = s.replace("²", "^2")
        s = s.replace("³", "^3")
        # Parse power
        power_count = unit.count("/")
        if power_count == 0:
            # Only numerator
            self.unit = s
            self.power = 1
        elif power_count == 1:
            unit_str, _, power_str = unit.partition("/")
            self.unit = unit_str
            self.power = int(power_str)
        else:
            raise ValueError(f"Unit '{unit}' contains more than one slash. Can't process !")
    
    def __eq__(self, other):
        if not isinstance(other, SubUnit):
            raise NotImplemented
        return self.unit == other.unit and self.power == other.power


class Unit(object):
    """
    Represents a potentially compound unit.
    For example, this class can 
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
        # Sort !
        self.numerator.sort()

    @staticmethod
    def _split_unit_string(s):
        """
        Split a multiplied list of units
        """
        return [entry for entry in s.split("*")]

    def __repr__(self):
        pass # TODO

    def __mul__(self, other):
        """
        Multiply this unit with another Unit.
        """
        if not isinstance(other, Unit):
            raise ValueError(f"Can't multiply a Unit with a non-Unit like {other}")
        pass # TODO



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
        return unit.unit
    except KeyError: # No return annotation
        raise UnannotatedReturnValueError(
            "Function {} does not have an annotated return value".format(fn))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Units, quantities and related"""
from collections import namedtuple
import functools

__all__ = ["Unit", "UnannotatedReturnValueError",
           "InvalidUnitInContextException", "InvalidUnitCombinationException",
           "find_returned_unit", "UnknownUnitInContextException",
           "Hz", "Pa", "rpm", "bar", "m", "g"]

Unit = namedtuple("Unit", ["unit"])

# Common unit constants for type annotations
Hz = Unit("Hz")
Pa = Unit("Pa")
rpm = Unit("rpm")
bar = Unit("bar")
m = Unit("m")
g = Unit("g")

class UnannotatedReturnValueError(Exception):
    """Raised if the automatic unit finder cannot find
    the appropriate function annotation that
    tells an auto-formatting function which unit is being used.

    Returns the unit string.
    """


class InvalidUnitInContextException(ValueError):
    """Raised if the unit might not be a globally unknown or invalid unit, but in the given context it can't be used."""


class UnknownUnitInContextException(ValueError):
    """Raised if the unit is not known in this context.

    The message should contain information on what type of
    qua
    ntity (e.g. length) is accepted.
    """

class InvalidUnitCombinationException(ValueError):
    """Raised if the units involved in an operation can't be
    combined in the way requested, for example if the
    user tries to add a voltage and a current.
    """

def find_returned_unit(fn):
    """Given a function that is assumed to return a quantity
    and annotated with the corresponding unit, determines
    which is the unit returned by the function
    """ if not callable(fn):     raise ValueError("fn must be callable".
)
    # Access innermost function inside possibly nested partials
    annotatedFN = fn
    while isinstance(annotatedFN, functools.partial):
        annotatedFN = annotatedFN.func
    # We have the innermost function
    try:
        unit = annotatedFN.__annotations__["return"]
        # Assume it's a Unit namedtuple
        return unit.unit
    except KeyError as exc: # No return annotation
        raise UnannotatedReturnValueError(
            f"Function {fn} does not have an annotated return value"
        ) from exc

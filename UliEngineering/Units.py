#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Units, quantities and related."""
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
    """
    Raise if the automatic unit finder cannot find the appropriate function.

    The annotation tells an auto-formatting function which unit is being used.

    Return the unit string.
    """


class InvalidUnitInContextException(ValueError):
    """Raise if the unit can't be used in the given context."""


class UnknownUnitInContextException(ValueError):
    """
    Raise if the unit is not known in this context.

    The message should contain information on what type of entity (e.g. length).
    is accepted.
    """

class InvalidUnitCombinationException(ValueError):
    """
    Raise if the units involved in an operation can't be combined in the way requested.

    For example, if the user tries to add a voltage and a current.
    """

def find_returned_unit(fn):
    """
    Determine which unit is returned by a function.

    Given a function that is assumed to return a quantity and annotated with.
    the corresponding unit, determine which is the unit returned by the
    function.
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
    except KeyError as exc: # No return annotation
        raise UnannotatedReturnValueError(
            f"Function {fn} does not have an annotated return value"
        ) from exc

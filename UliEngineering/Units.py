#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Units, quantities and related
"""
from collections import namedtuple
import functools

__all__ = ["Unit", "UnannotatedReturnValueError",
           "InvalidUnitInContextException",
           "find_returned_unit"]

Unit = namedtuple("Unit", ["unit"])

class UnannotatedReturnValueError(Exception):
    """
    Raised if the automatic unit finder cannot find
    the appropriate function annotation that
    tells an auto-formatting function which unit is being used.

    Returns the unit string
    """
    pass


class InvalidUnitInContextException(Exception):
    """
    Raised if the unit might not be a globally
    unknown or invalid unit, but in the given context
    it can't be used
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
            "Function {0} does not have an annotated return value")

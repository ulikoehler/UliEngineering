#!/usr/bin/env python3
"""
Specialized exceptions for UliEngineering
"""

__all__ = ["ConversionException", "InvalidUnitException",
           "OperationImpossibleException"]

class ConversionException(Exception):
    pass

class InvalidUnitException(ConversionException):
    pass

class OperationImpossibleException(Exception):
    """
    Raised if operation with the given parameters is impossible,
    i.e. they have the correct forward but the given application
    can't work with this specific set of values.
    """
    pass

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

class EngineerIOException(ValueError):
    """
    Baseclass for more specific EngineerIO exceptions
    """
    pass


class FirstCharacterInStringIsUnitPrefixException(EngineerIOException):
    """
    Raised if during parsing, multiple SI prefixes are detected, such as in pfJ,
    (pico-femto-Joules?!?)
    """
    pass
    
class MultipleUnitPrefixesException(EngineerIOException):
    """
    Raised if during parsing, multiple SI prefixes are detected, such as in pfJ,
    (pico-femto-Joules?!?)
    """
    pass


class RemainderOfStringContainsNonNumericCharacters(EngineerIOException):
    """
    Raised if, after stripping prefix, unit etc, off the string,
    there are still non-numeric characters left in the string
    """
    pass

class UnknownUnitInContextException(EngineerIOException):
    """
    Raised when an unknown unit is encountered in a specific context
    where only certain units are expected.
    """
    pass
#!/usr/bin/env python3
"""Specialized exceptions for UliEngineering."""

__all__ = ["ConversionException", "InvalidUnitException",
           "OperationImpossibleException"]

class ConversionException(Exception):
    """Exception raised for conversion errors."""

    pass

class InvalidUnitException(ConversionException):
    """Exception raised for invalid unit specifications."""

    pass

class OperationImpossibleException(Exception):
    """
    Raised if operation with the given parameters is impossible.

    i.e. they have the correct forward but the given application
    can't work with this specific set of values.
    """

    pass

class EngineerIOException(ValueError):
    """Base class for more specific EngineerIO exceptions."""

    pass


class FirstCharacterInStringIsUnitPrefixException(EngineerIOException):
    """
    Raise if multiple SI prefixes are detected during parsing.

    For example, in pfJ (pico-femto-Joules?!?).
    """

    pass

class MultipleUnitPrefixesException(EngineerIOException):
    """
    Raise if multiple SI prefixes are detected during parsing.

    For example, in pfJ (pico-femto-Joules?!?).
    """

    pass


class RemainderOfStringContainsNonNumericCharacters(EngineerIOException):
    """
    Raise if non-numeric characters remain after stripping prefix and unit.

    This occurs when, after stripping prefix, unit etc, off the string, there are
    still non-numeric characters left in the string.
    """

    pass

class UnknownUnitInContextException(EngineerIOException):
    """
    Raise when an unknown unit is encountered in a specific context.

    This occurs when only certain units are expected.
    """

    pass
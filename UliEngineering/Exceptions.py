#!/usr/bin/env python3
"""
Specialized exceptions for UliEngineering
"""

class ConversionException(Exception):
    pass

class InvalidUnitException(ConversionException):
    pass
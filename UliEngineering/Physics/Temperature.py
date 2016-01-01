#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding temperatures
"""
from UliEngineering.EngineerIO import normalizeEngineerInput
from UliEngineering.Exceptions import InvalidUnitException, ConversionException

zero_point_celsius = 273.15

def celsius_to_kelvin(c):
    return c + 273.15

def fahrenheit_to_kelvin(f):
    return (f + 459.67) * 5.0 / 9.0

def normalize_temperature(t, default_unit="°C"):
    """
    Normalize a temperature to kelvin.
    If it is a number or it has no unit, assume it is a default unit
    Else, evaluate the unit(K, °C, °F, C, F)
    """
    unit = ""
    if isinstance(t, str):
        res = normalizeEngineerInput(t)
        if res is None:
            raise ConversionException("Invalid temperature string: {0}".format(t))
        t, unit = res
    if not unit:
        unit = default_unit
    # Evaluate unit
    if unit in ["°C", "C"]:
        return celsius_to_kelvin(t)
    elif unit in ["°K", "K"]:
        return t
    elif unit in ["°F", "F"]:
        return fahrenheit_to_kelvin(t)
    else:
        raise InvalidUnitException("Unknown temperature unit: '{0}'".format(unit))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding temperatures
"""
from UliEngineering.EngineerIO import normalize_engineer_notation
from UliEngineering.Units import Unit
from UliEngineering.Exceptions import InvalidUnitException

try:
    from scipy.constants import zero_Celsius
except:
    zero_Celsius = 273.15 # Defined to that value

__all__ = ["celsius_to_kelvin", "kelvin_to_celsius",
           "fahrenheit_to_kelvin", "normalize_temperature",
           "normalize_temperature_celsius"]


def celsius_to_kelvin(c) -> Unit("°K"):
    return c + zero_Celsius


def kelvin_to_celsius(c) -> Unit("°C"):
    return c - zero_Celsius


def fahrenheit_to_kelvin(f) -> Unit("K"):
    return (f + 459.67) * 5.0 / 9.0


def normalize_temperature(t, default_unit="°C") -> Unit("°K"):
    """
    Normalize a temperature to kelvin.
    If it is a number or it has no unit, assume it is a default unit
    Else, evaluate the unit(K, °C, °F, C, F)
    """
    unit = ""
    if isinstance(t, str):
        res = normalize_engineer_notation(t)
        if res is None:
            raise ValueError("Invalid temperature string: {0}".format(t))
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


def normalize_temperature_celsius(t, default_unit="°C") -> Unit("°C"):
    """Like normalize_temperature(), but returns a value in celsius instead of Kelvin"""
    return kelvin_to_celsius(normalize_temperature(t, default_unit))

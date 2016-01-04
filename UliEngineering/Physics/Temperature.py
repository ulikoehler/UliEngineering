#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding temperatures
"""
from UliEngineering.EngineerIO import normalizeEngineerInput, Quantity
from UliEngineering.Exceptions import InvalidUnitException, ConversionException

"""Celsius zero in Kelvin"""
zero_point_celsius = 273.15

def celsius_to_kelvin(c) -> Quantity("°K"):
    return c + zero_point_celsius

def kelvin_to_celsius(c) -> Quantity("°C"):
    return c - zero_point_celsius

def fahrenheit_to_kelvin(f) -> Quantity("K"):
    return (f + 459.67) * 5.0 / 9.0

def normalize_temperature(t, default_unit="°C") -> Quantity("°K"):
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

def normalize_temperature_celsius(t, default_unit="°C") -> Quantity("°C"):
    """Like normalize_temperature(), but returns a value in celsius instead of Kelvin"""
    return kelvin_to_celsius(normalize_temperature(t, default_unit))

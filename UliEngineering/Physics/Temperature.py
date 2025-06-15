#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding temperatures
"""
from UliEngineering.EngineerIO import normalize
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from UliEngineering.Exceptions import InvalidUnitException

try:
    from scipy.constants import zero_Celsius
except:
    zero_Celsius = 273.15 # Defined constant for 0 °C in Kelvin

__all__ = ["celsius_to_kelvin", "kelvin_to_celsius",
           "fahrenheit_to_kelvin", "normalize_temperature",
           "normalize_temperature_celsius",
           "normalize_temperature_kelvin",
           "temperature_with_dissipation",
           "fahrenheit_to_celsius", "zero_Celsius"]

@returns_unit("K")
@normalize_numeric_args
def celsius_to_kelvin(c):
    return c + zero_Celsius

@returns_unit("°C")
@normalize_numeric_args
def kelvin_to_celsius(c):
    return c - zero_Celsius

@returns_unit("K")
@normalize_numeric_args
def fahrenheit_to_kelvin(f):
    return (f + 459.67) * 5.0 / 9.0

@returns_unit("°C")
def fahrenheit_to_celsius(f):
    return kelvin_to_celsius(fahrenheit_to_kelvin(f))

@returns_unit("K")
def normalize_temperature(t, default_unit="°C"):
    """
    Normalize a temperature to kelvin.
    If it is a number or it has no unit, assume it is a default unit
    Else, evaluate the unit(K, °C, °F, C, F)
    """
    unit = ""
    if isinstance(t, str):
        res = normalize(t)
        if res is None:
            raise ValueError("Invalid temperature string: {}".format(t))
        t, unit = res.value, res.unit
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
        raise InvalidUnitException("Unknown temperature unit: '{}'".format(unit))

normalize_temperature_kelvin = normalize_temperature

@returns_unit("°C")
def normalize_temperature_celsius(t, default_unit="°C"):
    """Like normalize_temperature(), but returns a value in celsius instead of Kelvin"""
    return kelvin_to_celsius(normalize_temperature(t, default_unit))

@returns_unit("°C")
@normalize_numeric_args
def temperature_with_dissipation(power_dissipated="1 W", theta="50 °C/W", t_ambient="25 °C"):
    """
    Compute the temperature of a component, given its thermal resistance (theta),
    its dissipated power and 
    """
    t_ambient = normalize_temperature_celsius(t_ambient)
    return t_ambient + power_dissipated * theta

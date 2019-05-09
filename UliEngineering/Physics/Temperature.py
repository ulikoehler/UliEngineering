#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding temperatures
"""
from UliEngineering.EngineerIO import normalize_engineer_notation, normalize_numeric
from UliEngineering.Units import Unit
from UliEngineering.Exceptions import InvalidUnitException

try:
    from scipy.constants import zero_Celsius
except:
    zero_Celsius = 273.15 # Defined to that value

__all__ = ["celsius_to_kelvin", "kelvin_to_celsius",
           "fahrenheit_to_kelvin", "normalize_temperature",
           "normalize_temperature_celsius",
           "temperature_with_dissipation",
           "fahrenheit_to_celsius"]

def celsius_to_kelvin(c) -> Unit("K"):
    c = normalize_numeric(c)
    return c + zero_Celsius

def kelvin_to_celsius(c) -> Unit("°C"):
    c = normalize_numeric(c)
    return c - zero_Celsius

def fahrenheit_to_kelvin(f) -> Unit("K"):
    f = normalize_numeric(f)
    return (f + 459.67) * 5.0 / 9.0

def fahrenheit_to_celsius(f) -> Unit("°C"):
    return kelvin_to_celsius(fahrenheit_to_kelvin(f))

def normalize_temperature(t, default_unit="°C") -> Unit("K"):
    """
    Normalize a temperature to kelvin.
    If it is a number or it has no unit, assume it is a default unit
    Else, evaluate the unit(K, °C, °F, C, F)
    """
    unit = ""
    if isinstance(t, str):
        res = normalize_engineer_notation(t)
        if res is None:
            raise ValueError("Invalid temperature string: {}".format(t))
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
        raise InvalidUnitException("Unknown temperature unit: '{}'".format(unit))

def normalize_temperature_celsius(t, default_unit="°C") -> Unit("°C"):
    """Like normalize_temperature(), but returns a value in celsius instead of Kelvin"""
    return kelvin_to_celsius(normalize_temperature(t, default_unit))

def temperature_with_dissipation(power_dissipated="1 W", theta="50 °C/W", t_ambient="25 °C") -> Unit("°C"):
    """
    Compute the temperature of a component, given its thermal resistance (theta),
    its dissipated power and 
    """
    power_dissipated = normalize_numeric(power_dissipated)
    theta = normalize_numeric(theta)
    t_ambient = normalize_temperature_celsius(t_ambient)
    return t_ambient + power_dissipated * theta

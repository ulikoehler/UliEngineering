#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities regarding temperatures."""
from typing import Annotated

from UliEngineering.EngineerIO import normalize, normalize_numeric
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Exceptions import InvalidUnitException

try:
    from scipy.constants import zero_Celsius
except ImportError:
    zero_Celsius = 273.15 # Defined constant for 0 °C in Kelvin

__all__ = ["celsius_to_kelvin", "kelvin_to_celsius",
           "fahrenheit_to_kelvin", "normalize_temperature",
           "normalize_temperature_celsius",
           "normalize_temperature_kelvin",
           "temperature_with_dissipation",
           "fahrenheit_to_celsius", "zero_Celsius",
           "TemperatureKelvin", "TemperatureCelsius"]

@returns_unit("K")
def celsius_to_kelvin(c: NormalizableArgument):
    c = normalize_numeric(c) if isinstance(c, str) else c
    return c + zero_Celsius

@returns_unit("°C")
def kelvin_to_celsius(c: NormalizableArgument):
    c = normalize_numeric(c) if isinstance(c, str) else c
    return c - zero_Celsius

@returns_unit("K")
def fahrenheit_to_kelvin(f: NormalizableArgument):
    f = normalize_numeric(f) if isinstance(f, str) else f
    return (f + 459.67) * 5.0 / 9.0

@returns_unit("°C")
def fahrenheit_to_celsius(f: NormalizableArgument):
    f = normalize_numeric(f) if isinstance(f, str) else f
    return kelvin_to_celsius(fahrenheit_to_kelvin(f))

@returns_unit("K")
def normalize_temperature(t: NormalizableArgument, default_unit="°C") -> NormalizedComputable:
    """
    Normalize a temperature to kelvin.

    If it is a number or it has no unit, assume it is a default unit. Else,.
    evaluate the unit (K, °C, °F, C, F).
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
def normalize_temperature_celsius(t: NormalizableArgument, default_unit="°C") -> NormalizedComputable:
    """
    Normalize a temperature to celsius.

    Like normalize_temperature(), but returns a value in celsius instead of.
    Kelvin.
    """
    return kelvin_to_celsius(normalize_temperature(t, default_unit))


# Unit type annotations
TemperatureKelvin = Annotated[NormalizedComputable, normalize_temperature]
TemperatureCelsius = Annotated[NormalizedComputable, normalize_temperature_celsius]

@returns_unit("°C")
def temperature_with_dissipation(power_dissipated: NormalizableArgument = "1 W", theta: NormalizableArgument = "50 °C/W", t_ambient: NormalizableArgument = "25 °C"):
    """
    Compute the temperature of a component given its thermal resistance.
    
    dissipated power, and ambient temperature.
    """
    power_dissipated = normalize_numeric(power_dissipated) if isinstance(power_dissipated, str) else power_dissipated
    theta = normalize_numeric(theta) if isinstance(theta, str) else theta
    t_ambient = normalize_temperature_celsius(t_ambient)
    return t_ambient + power_dissipated * theta

#!/usr/bin/env python3
from Resistors import *
import scipy.constants
from EngineerIO import *
import math

class ConversionException(Exception):
    pass

def celsius_to_kelvin(c):
    return c + 273.15

def fahrenheit_to_kelvin(f):
    return (f + 459.67) * 5.0/9.0

def normalize_temperature(t, default_unit="°C"):
    """
    Normalize a temperature to kelvin.
    If it is a number or it has no unit, assume it is a default unit
    Else, evaluate the unit(K, °C, °F, C, F)

    TODO: Support degree sign

    >>> normalize_temperature("0")
    273.15
    >>> normalize_temperature("1")
    274.15
    >>> normalize_temperature("1 C")
    274.15
    >>> normalize_temperature("1 K")
    1.0
    >>> "%.2f" % normalize_temperature("60 F")
    '288.71'
    """
    unit = ""
    if isinstance(t, str):
        t, unit = normalizeEngineerInput(t)
    if not unit:
        unit = default_unit
    #Evaluate unit
    if unit == "°C" or unit == "C":
        return celsius_to_kelvin(t)
    elif unit == "°K" or unit == "K":
        return t
    elif unit == "F" or unit == "°F":
        return fahrenheit_to_kelvin(t)
    else:
        raise ConversionException("Unknown temperature unit: '%s'" % unit)

def johnson_nyquist_noise_current(r, delta_f, T):
    """
    Compute the Johnson Nyquist noise current in amperes
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    if isinstance(r, str):
        r, _ = normalizeEngineerInput(r)
    if isinstance(delta_f, str):
        delta_f, _ = normalizeEngineerInput(delta_f)
    t_kelvin = normalize_temperature(T)
    #Support celsius and kelvin inputs
    return math.sqrt((4 * scipy.constants.k * t_kelvin * delta_f)/r)

def johnson_nyquist_noise_voltage(r, delta_f, T):
    """
    Compute the Johnson Nyquist noise current in amperes
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    if isinstance(r, str):
        r, _ = normalizeEngineerInput(r)
    if isinstance(delta_f, str):
        delta_f, _ = normalizeEngineerInput(delta_f)
    t_kelvin = celsius_to_kelvin(T)
    return math.sqrt(4 * scipy.constants.k * t_kelvin * delta_f * r)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Usage example
    print(formatValue(johnson_nyquist_noise_current("20 MΩ", 1000, 25), "A"))
    print(formatValue(johnson_nyquist_noise_voltage("10 MΩ", 1000, 25), "V"))

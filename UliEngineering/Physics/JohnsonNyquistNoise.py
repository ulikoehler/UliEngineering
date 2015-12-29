#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Johnson Nyquist noise utilities for both voltage and current noise
"""
import scipy.constants
from .Resistors import *
from .EngineerIO import *
import math

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
    if isinstance(T, str):
        T, _ = normalizeEngineerInput(T)
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
    if isinstance(T, str):
        T, _ = normalizeEngineerInput(T)
    t_kelvin = celsius_to_kelvin(T)
    return math.sqrt(4 * scipy.constants.k * t_kelvin * delta_f * r)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Usage example
    print(formatValue(johnson_nyquist_noise_current("20 MΩ", 1000, 25), "A"))
    print(formatValue(johnson_nyquist_noise_voltage("10 MΩ", 1000, 25), "V"))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Johnson Nyquist noise utilities for both voltage and current noise

# Usage example
>>> from UliEngineering.Physics.JohnsonNyquistNoise import *
>>> from UliEngineering.EngineerIO import autoFormat
>>> print(autoFormat(johnson_nyquist_noise_current, "20 MΩ", 1000, "20 °C"))
>>> print(autoFormat(johnson_nyquist_noise_voltage, "10 MΩ", 1000, 25))
"""
import scipy.constants
from .Resistors import *
from .Temperature import normalize_temperature, celsius_to_kelvin
from UliEngineering.EngineerIO import normalizeEngineerInputIfStr, Quantity
import math

def johnson_nyquist_noise_current(r, delta_f, T) -> Quantity("A"):
    """
    Compute the Johnson Nyquist noise current in amperes
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    r, _ = normalizeEngineerInputIfStr(r)
    delta_f, _ = normalizeEngineerInputIfStr(delta_f)
    t_kelvin = normalize_temperature(T)
    # Support celsius and kelvin inputs
    return math.sqrt((4 * scipy.constants.k * t_kelvin * delta_f)/r)

def johnson_nyquist_noise_voltage(r, delta_f, T) -> Quantity("V"):
    """
    Compute the Johnson Nyquist noise current in amperes
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    r, _ = normalizeEngineerInputIfStr(r)
    delta_f, _ = normalizeEngineerInputIfStr(delta_f)
    t_kelvin = normalize_temperature(T)
    return math.sqrt(4 * scipy.constants.k * t_kelvin * delta_f * r)

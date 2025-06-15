#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Johnson Nyquist noise utilities for both voltage and current noise

# Usage example
>>> from UliEngineering.Physics.JohnsonNyquistNoise import *
>>> from UliEngineering.EngineerIO import autoFormat
>>> print(autoFormat(johnson_nyquist_noise_current, "20 MΩ", 1000, "20 °C"))
>>> print(autoFormat(johnson_nyquist_noise_voltage, "10 MΩ", 1000, 25))
"""
from .Temperature import normalize_temperature
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import math

__all__ = ["johnson_nyquist_noise_current", "johnson_nyquist_noise_voltage"]

try:
    from scipy.constants import k as boltzmann_k
except ModuleNotFoundError:
    # Exact defined value: https://physics.nist.gov/cgi-bin/cuu/Value?k
    boltzmann_k = 1.380649e-23

@returns_unit("A")
@normalize_numeric_args
def johnson_nyquist_noise_current(r, delta_f, T):
    """
    Compute the Johnson Nyquist noise current in amperes
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    t_kelvin = normalize_temperature(T)
    # Support celsius and kelvin inputs
    return math.sqrt((4 * boltzmann_k * t_kelvin * delta_f)/r)

@returns_unit("V")
@normalize_numeric_args
def johnson_nyquist_noise_voltage(r, delta_f, T):
    """
    Compute the Johnson Nyquist noise voltage in volts
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    t_kelvin = normalize_temperature(T)
    return math.sqrt(4 * boltzmann_k * t_kelvin * delta_f * r)

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
from .Temperature import normalize_temperature
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import math

__all__ = ["johnson_nyquist_noise_current", "johnson_nyquist_noise_voltage"]

try:
    from scipy.constants import k as boltzmann_k
except ModuleNotFoundError:
    # Accept and use hardcoded but warn
    import warnings
    warnings.warn("Using hardcoded Boltzmann k_b constant value. This is a measured quantity! Installing scipy is recommended!", RuntimeWarning)
    boltzmann_k = 1.38064852e-23 # Recommended value as of 2015

def johnson_nyquist_noise_current(r, delta_f, T) -> Unit("A"):
    """
    Compute the Johnson Nyquist noise current in amperes
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    r = normalize_numeric(r)
    delta_f = normalize_numeric(delta_f)
    t_kelvin = normalize_temperature(T)
    # Support celsius and kelvin inputs
    return math.sqrt((4 * boltzmann_k * t_kelvin * delta_f)/r)


def johnson_nyquist_noise_voltage(r, delta_f, T) -> Unit("V"):
    """
    Compute the Johnson Nyquist noise voltage in volts
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts
    """
    r = normalize_numeric(r)
    delta_f = normalize_numeric(delta_f)
    t_kelvin = normalize_temperature(T)
    return math.sqrt(4 * boltzmann_k * t_kelvin * delta_f * r)

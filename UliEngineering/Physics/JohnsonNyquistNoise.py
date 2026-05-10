#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Johnson Nyquist noise utilities for both voltage and current noise.

Usage example:
>>> from UliEngineering.Physics.JohnsonNyquistNoise import *
>>> from UliEngineering.EngineerIO import autoFormat
>>> print(autoFormat(johnson_nyquist_noise_current, "20 MΩ", 1000, "20 °C"))
>>> print(autoFormat(johnson_nyquist_noise_voltage, "10 MΩ", 1000, 25))
"""
from typing import Annotated

from .Temperature import normalize_temperature, TemperatureKelvin
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics.Frequency import FrequencyHz, normalize_frequency
from ._normalize import normalize_with_known_units
import math

__all__ = ["johnson_nyquist_noise_current", "johnson_nyquist_noise_voltage",
           "normalize_resistance", "ResistanceOhm",
           "normalize_temperature", "TemperatureKelvin",
           "normalize_frequency", "FrequencyHz"]

try:
    from scipy.constants import k as boltzmann_k
except ModuleNotFoundError:
    # Exact defined value: https://physics.nist.gov/cgi-bin/cuu/Value?k
    boltzmann_k = 1.380649e-23

def normalize_resistance(resistance: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(resistance, {"Ω": 1.0, "Ohm": 1.0, "ohm": 1.0, "R": 1.0, "kΩ": 1000.0, "MΩ": 1e6, "GΩ": 1e9, "mΩ": 1e-3, "µΩ": 1e-6}, quantity_name="resistance")

ResistanceOhm = Annotated[NormalizedComputable, normalize_resistance]

@returns_unit("A")
def johnson_nyquist_noise_current(r: ResistanceOhm, delta_f: FrequencyHz, T: TemperatureKelvin):
    """
    Compute the Johnson Nyquist noise current in amperes
    T must be given in °C whereas r must be given in Ohms.
    The result is given in volts

    """
    r = normalize_resistance(r)
    delta_f = normalize_frequency(delta_f)
    t_kelvin = normalize_temperature(T)
    # Support celsius and kelvin inputs
    return math.sqrt((4 * boltzmann_k * t_kelvin * delta_f)/r)

@returns_unit("V")
def johnson_nyquist_noise_voltage(r: ResistanceOhm, delta_f: FrequencyHz, T: TemperatureKelvin):
    """
    Compute the Johnson Nyquist noise voltage in volts
    T must be given in °C whereas r must be given in Ohms.

    The result is given in volts
    """
    r = normalize_resistance(r)
    delta_f = normalize_frequency(delta_f)
    t_kelvin = normalize_temperature(T)
    return math.sqrt(4 * boltzmann_k * t_kelvin * delta_f * r)

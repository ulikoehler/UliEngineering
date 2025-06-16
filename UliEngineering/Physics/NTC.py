#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding NTC thermistors

See http://www.vishay.com/docs/29053/ntcintro.pdf for details
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Physics.Temperature import normalize_temperature
from UliEngineering.EngineerIO.Decorators import returns_unit
import numpy as np
from .Temperature import normalize_temperature_celsius, zero_Celsius

__all__ = ["ntc_resistance", "ntc_resistances"]

@returns_unit("Ω")
def ntc_resistance(r25, b25, t):
    """
    Compute the NTC resistance by temperature and NTC parameters

    Parameters
    ----------
    r25 : float or EngineerIO string
        The NTC resistance at 25°C, sometimes also called "nominal resistance"
    b25: float or EngineerIO string
        The NTC b-constant (e.g. b25/50, b25/85 or b25/100)
    t : temperature
        The temperature. Will be interpreted using normalize_temperature()
    """
    r25 = normalize_numeric(r25)
    b25 = normalize_numeric(b25)
    t = normalize_temperature(t)  # t is now in Kelvins
    # Compute resistance
    return r25 * np.exp(b25 * (1.0 / t - 1.0 / (25.0 + zero_Celsius)))

@returns_unit("Ω")
def ntc_resistances(r25, b25, t0=-40, t1=85, resolution=0.1):
    """
    Compute the resistances over a temperature range with a given resolution.

    Returns
    =======
    A (temperatures, values) tuple
    """
    r25 = normalize_numeric(r25)
    b25 = normalize_numeric(b25)
    # Convert all temperatures to Celsius
    t0 = normalize_temperature_celsius(t0)
    t1 = normalize_temperature_celsius(t1)
    resolution = normalize_temperature_celsius(resolution)
    # Create a temperature range
    ts = np.linspace(t0, t1, int((t1 - t0) // resolution + 1))
    return ts, ntc_resistance(r25, b25, ts)

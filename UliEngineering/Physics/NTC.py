#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding NTC thermistors

See http://www.vishay.com/docs/29053/ntcintro.pdf for details
"""
from UliEngineering.Physics.Temperature import normalize_temperature
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np
from .Temperature import zero_Celsius

__all__ = ["ntc_resistance", "ntc_resistances"]


def ntc_resistance(r25, b25, t) -> Unit("Ω"):
    """
    Compute the NTC resistance by  temperature and NTC parameters

    Parameters
    ----------
    r25 : float or EngineerIO string
        The NTC resistance at 25°C, sometimes also called "nominal resistance"
    b25: float or EngineerIO string
        The NTC b-constant (e.g. b25/50, b25/85 or b25/100)
    t : temperature
        The temperature. Will be interpreted using normalize_temperature()
    """
    # Normalize inputs
    r25 = normalize_numeric(r25)
    b25 = normalize_numeric(b25)
    t = normalize_temperature(t) # t is now in Kelvins
    # Compute resistance
    return r25 * np.exp(b25 * (1./t - 1./(25. + zero_Celsius)))

def ntc_resistances(r25, b25, t0=-40, t1=85, resolution=0.1):
    """
    Compute the resistances over a temperature range with a given resolution.

    Returns
    =======
    A (temperatures, values) tuple
    """
    ts = np.linspace(t0, t1, int((t1 - t0) // resolution + 1))
    return ts, ntc_resistance(r25, b25, ts)

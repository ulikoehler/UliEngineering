#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding NTC thermistors

See http://www.vishay.com/docs/29053/ntcintro.pdf for details
"""
from UliEngineering.Physics.Temperature import zero_point_celsius, normalize_temperature
from UliEngineering.EngineerIO import normalize_numeric, Quantity
import functools
from collections import namedtuple
import numpy as np


def ntc_resistance(r25, b25, t) -> Quantity("Ω"):
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
    t = normalize_temperature(t)
    # Compute resistance
    return r25 * np.exp(b25 * (1./t - 1./(25. + zero_point_celsius)))

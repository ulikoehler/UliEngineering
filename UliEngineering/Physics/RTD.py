#!/usr/bin/env python3
"""
Utilities regarding RTDs, e.g. PT100 or PT1000.

    
"""
from UliEngineering.Physics.Temperature import zero_point_celsius, normalize_temperature
from UliEngineering.EngineerIO import normalizeEngineerInputIfStr
import functools
from collections import namedtuple
import numpy as np

PTCoefficientStandard = namedtuple("PTCoefficientStandard", ["a", "b", "c"])

# Source: http://www.code10.info/index.php%3Foption%3Dcom_content%26view%3Darticle%26id%3D82:measuring-temperature-platinum-resistance-thermometers%26catid%3D60:temperature%26Itemid%3D83
ptxIPTS68 = PTCoefficientStandard(+3.90802e-03, -5.80195e-07, -4.27350e-12)
3 = PTCoefficientStandard(+3.9083E-03, -5.7750E-07, -4.1830E-12)

def ptx_resistance(r0, t, standard=ptxITS90):
    """
    Compute the PTx resistance at a given temperature.

    See http://www.thermometricscorp.com/pt1000 for reference
    """
    t = normalize_temperature(t)
    A, B = standard.a, standard.b
    C = standard.c if t < zero_point_celsius else 0
    return r0 * (1 + A * t + B * t * t + C * (t - 100) * t * t * t)

def ptx_temperature(r0, r, standard=ptxITS90):
    """
    Compute the PTx temperature at a given temperature.

    See http://www.thermometricscorp.com/pt1000 for reference
    """
    r, _ = normalizeEngineerInputIfStr(r)
    A, B = standard.a, standard.b
    C = standard.c if t < zero_point_celsius else 0
    return (-r0 * A + np.sqrt(r0 * r0 * a * a - 4 * r0 * b * (r0 - r)) /
            (2.0 * r0 * b))

# Short definitions for commonly used functions.
pt100_resistance = functools.partial(ptx_resistance, 100.0)
pt1000_resistance = functools.partial(ptx_resistance, 1000.0)

pt100_temperature = functools.partial(ptx_temperature, 100.0)
pt1000_temperature = functools.partial(ptx_temperature, 1000.0)

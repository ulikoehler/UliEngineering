#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding RTDs, e.g. PT100 or PT1000.

=====
PT1000 Rationale

This module implements a polynomial-fit based correction to obtain mathematically accurate
temperature values (down to < 58.6 µ°C) for PT100/PT1000 given a measured resistance.

The correction polynomial is precalculated and automatically applied if r0==100.0 or r0==1000.0.
For other r0 values, you can precalculate the polynomial.

For details read:
https://techoverflow.net/blog/2016/01/02/accurate-calculation-of-pt100-pt1000-temperature-from-resistance/
"""
from typing import Annotated

from UliEngineering.Physics.Temperature import normalize_temperature_celsius, TemperatureKelvin
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ._normalize import normalize_with_known_units
import functools
from collections import namedtuple
import numpy as np
import numbers

__all__ = [
    "ptx_resistance", "ptx_temperature",
    "check_correction_polynomial_quality", "compute_correction_polynomial",
    "pt100_resistance", "pt1000_resistance",
    "pt100_temperature", "pt1000_temperature",
    "normalize_resistance", "ResistanceOhm",
    "normalize_temperature_celsius", "TemperatureKelvin"
]

def normalize_resistance(resistance: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(resistance, {"Ω": 1.0, "Ohm": 1.0, "ohm": 1.0, "R": 1.0,
                                                   "kΩ": 1000.0, "MΩ": 1e6, "GΩ": 1e9,
                                                   "mΩ": 1e-3, "µΩ": 1e-6, "k": 1000.0,
                                                   "M": 1e6, "G": 1e9},
                                      quantity_name="resistance")


ResistanceOhm = Annotated[NormalizedComputable, normalize_resistance]

PTCoefficientStandard = namedtuple("PTCoefficientStandard", ["a", "b", "c"])

# Source: http://www.code10.info/index.php%3Foption%3Dcom_content%26view%3Darticle%26id%3D82:measuring-temperature-platinum-resistance-thermometers%26catid%3D60:temperature%26Itemid%3D83
ptx_ipts68 = PTCoefficientStandard(+3.90802e-03, -5.80195e-07, -4.27350e-12)
ptx_its90 = PTCoefficientStandard(+3.9083E-03, -5.7750E-07, -4.1830E-12)

no_correction = np.poly1d([])
pt1000_correction = np.poly1d([1.51892983e-15, -2.85842067e-12, -5.34227299e-09,
                              1.80282972e-05, -1.61875985e-02, 4.84112370e+00])
pt100_correction = np.poly1d([1.51892983e-10, -2.85842067e-08, -5.34227299e-06,
                             1.80282972e-03, -1.61875985e-01, 4.84112370e+00])


@returns_unit("Ω")
def ptx_resistance(r0: ResistanceOhm, t: TemperatureKelvin, standard=ptx_its90):
    """
    Compute the PTx resistance at a given temperature.
    
    The reference for the test code is a DIN PT1000.

    See http://www.thermometricscorp.com/pt1000 for reference
    """
    r0 = normalize_resistance(r0)
    t = normalize_temperature_celsius(t)
    A, B = standard.a, standard.b
    # C := 0 for t > 0, else std.c. This also works for numpy arrays
    if isinstance(t, numbers.Number):
        C = standard.c if t < 0.0 else 0
    else:
        C = np.piecewise(t, [t < 0, t >= 0], [standard.c, 0])
    return r0 * (1.0 + A * t + B * t * t + C * (t - 100.0) * t * t * t)


@returns_unit("°C")
def ptx_temperature(r0: ResistanceOhm, r: ResistanceOhm, standard=ptx_its90, poly=None):
    """
    Compute the PTx temperature at a given temperature.

    Accepts an additive correction polynomial that is applied to the resistance.
    If the poly kwarg is None, the polynom is automatically selected.
    no_correction is used for other r0 values. In this case, use a
    custom polynomial (numpy poly1d object) as the poly kwarg.

    See http://www.thermometricscorp.com/pt1000 for reference
    """
    r0 = normalize_resistance(r0)
    r = normalize_resistance(r)
    A, B = standard.a, standard.b
    # Select
    if poly is None:
        if abs(r0 - 1000.0) < 1e-3: poly = pt1000_correction
        elif abs(r0 - 100.0) < 1e-3: poly = pt100_correction
        else: poly = no_correction

    t = ((-r0 * A + np.sqrt(r0 * r0 * A * A - 4 * r0 * B * (r0 - r))) /
         (2.0 * r0 * B))
    # For subzero-temperature refine the computation by the correction polynomial
    if isinstance(r, numbers.Number):
        if r < r0:
            t += poly(r)
    else:  # Treated like a numpy array
        t += poly(r) * np.piecewise(r, [r < r0, r >= r0], [1.0, 0.0])
    return t


def check_correction_polynomial_quality(r0, reftemp, poly):
    """
    Get a difference array for a given correction polynomial.
    
    Return (resistances, diffarray, peak-to-peak scalar).
    """
    # Compute reftemp -> resistance -> computed temp
    resistances = ptx_resistance(r0, reftemp)
    temperatures = ptx_temperature(r0, resistances, poly=poly)
    tempdiff = reftemp - temperatures
    quality = np.max([np.abs(tempdiff.max()), np.abs(tempdiff.min())])
    return (resistances, tempdiff, quality)

def compute_correction_polynomial(r0, order=5, n=1000000) -> np.poly1d:
    """
    Compute a correction polynomial that can be applied to the resistance.
    
    to get an additive correction coefficient that approximately corrects.
    for errors induced by the C * (t - 100) * t³ term in the formula which
    can't be easily solved.

    This module contains several precomputed polynomials:
        - no_correction
        - pt1000_correction
        - pt100_correction

    It is recommended to use order=5 for this problem.
    """
    # Compute values with no correct
    reftemp = np.linspace(-200.0, 0.0, n)
    resistances, tempdiff, _ = check_correction_polynomial_quality(r0, reftemp, poly=no_correction)
    # Compute best polynomial
    return np.poly1d(np.polyfit(resistances, tempdiff, order))


# Short definitions for commonly used functions.
pt100_resistance = functools.partial(ptx_resistance, 100.0)
pt1000_resistance = functools.partial(ptx_resistance, 1000.0)

pt100_temperature = functools.partial(ptx_temperature, 100.0)
pt1000_temperature = functools.partial(ptx_temperature, 1000.0)

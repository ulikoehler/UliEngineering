#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing temperature coefficients
and their effects
"""
from UliEngineering.EngineerIO import normalize_numeric, normalize, format_value
from UliEngineering.Units import Unit
from collections import namedtuple

__all__ = ["ValueRange", "value_range_over_temperature"]

ValueRange = namedtuple("ValueRangeOverTemperature", ["min", "max"])

def value_range_over_temperature(nominal, coefficient="100ppm", tmin="-40 °C", tmax="85 °C", tnom="25 °C", significant_digits=4):
    """
    Given a component which has a nominal value (e.g. "1 kΩ")
    at tnom (typically "25 °C") and a coefficient of temperature (e.g. "100ppm").

    Computes the mininimum and maximum possible value of that component
    over the entire temperature range.
    
    The coefficient can be given e.g. as ppm, ppb or % (use a string!)
    and is interpreted as "plus-minus" value.
    Alternatively, it can be given as a 2-tuple indicating separate "+"
    and "-" values (e.g. ("0 ppm", "100 ppm")).

    Optionally, a component tolerance can be given (defaults to "0 %")
    to also account for static (temperature-independent) differences. 

    Usage example:

    Keyword arguments
    -----------------
    nominal : number or string
        The nominal value of the component e.g. 1023 or "1.023 kΩ"
    coefficient : number or string
        The temperature coefficient of the component per °C
        e.g. "100 ppm", "1 %" or 100e-6
    tmin : number of string
        The minimum temperature to consider in °C, e.g. "-40 °C". or -40.
    tmax : number of string
        The maximum temperature to consider in °C, e.g. "85 °C". or 85.
    tnom : number or string
        The temperature at which the nominal value was measured.
        If not specified in the component datasheet, this is usually "20 °C" or "25 °C".
    significant_digits : integer
        How many significant digits to show in the resulting value strings
    Returns
    -------
    A ValueRange() instance containing strings with the correct unit, if any.
    Example: ValueRange("99.5 Ω", "100.5 Ω")
    Use .min and .max to get the min/max value
    """
    tmin = normalize_numeric(tmin)
    tmax = normalize_numeric(tmax)
    tnom = normalize_numeric(tnom)
    # We are only interested in the maximum temperature
    # differential from tmin
    tdelta_max = max(abs(tnom - tmin), abs(tmax - tnom))
    nominal, unit = normalize(nominal)
    # Parse coefficient and compute min & max factors
    if isinstance(coefficient, tuple):
        # Check length 2
        if len(coefficient) != 2:
            raise ValueError("If coefficient is given as a tuple, it must have length 2. Coefficient is {}".format(coefficient))
        # Parse tuple
        neg_coefficient = normalize_numeric(coefficient[0])
        pos_coefficient = normalize_numeric(coefficient[1])
    else:
        coefficient = normalize_numeric(coefficient)
        pos_coefficient = coefficient
        neg_coefficient = coefficient
    
    pos_factor = 1. + (tdelta_max * pos_coefficient)
    neg_factor = 1. - (tdelta_max * neg_coefficient)
    # Min: Worst case minimum 
    min_value = neg_factor * nominal
    max_value = pos_factor * nominal

    return ValueRange(
        format_value(min_value, unit, significant_digits=significant_digits),
        format_value(max_value, unit, significant_digits=significant_digits)
    )
    

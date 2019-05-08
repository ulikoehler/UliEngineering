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

def _normalize_minmax_tuple(arg, name="field"):
    """
    Interprets arg either a single +- value or as
    a 2-tuple of + and - values.
    All vaues 

    If arg is a tuple:
        Return ValueRange(arg[0], arg[1]) (strings are normalized)
    Else:
        Return ValueRange(-arg, +arg) (strings are normalized)
    """
    # Parse coefficient and compute min & max factors
    if isinstance(arg, tuple):
        # Check length 2
        if len(arg) != 2:
            raise ValueError("If {} is given as a tuple, it must have length 2. {} is {}".format(name, name, arg))
        # Parse tuple
        min_value = normalize_numeric(arg[0])
        max_value = normalize_numeric(arg[1])
    else:
        arg = normalize_numeric(arg)
        min_value = -arg
        max_value = arg
    return ValueRange(min_value, max_value)

def value_range_over_temperature(nominal, coefficient="100ppm", tolerance="0 %", tmin="-40 °C", tmax="85 °C", tnom="25 °C", significant_digits=4):
    """
    Given a component which has a nominal value (e.g. "1 kΩ")
    at tnom (typically "25 °C") and a coefficient of temperature (e.g. "100ppm").

    Computes the mininimum and maximum possible value of that component
    over the entire temperature range.

    Optionally, a component tolerance can be given (defaults to "0 %")
    to also account for static (temperature-independent) differences. 
    Note that the tolerance is applied to the nominal value before
    applying the temperature coefficient.
    
    The coefficient and static tolerance can be given as number or as string
    e.g. as ppm, ppb or %
    and is interpreted as "plus-minus" value.
    Alternatively, it can be given as a 2-tuple indicating separate "+"
    and "-" values (e.g. ("-30 ppm", "100 ppm")).
    Note that the negative value has to be negative if there is a
    negative temperature coefficient!

    Usage example:

    Keyword arguments
    -----------------
    nominal : number or string
        The nominal value of the component e.g. 1023 or "1.023 kΩ"
    coefficient : number or string or 2-tuple (see above)
        The temperature coefficient of the component per °C
        e.g. "100 ppm", "1 %" or 100e-6
        or: ("-30 ppm", "100 ppm") (separate + and - values)
    tolerance : number or string or 2-tuple (see above)
        The static (temperature-independent) tolerance of the component.
        e.g. "100 ppm", "1 %" or 100e-6
        or: ("-0.5 %", "1.0%") (separate + and - values)
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
    # Static tolerance
    # We are only interested in the maximum temperature
    # differential from tmin
    tdelta_max = max(abs(tnom - tmin), abs(tmax - tnom))
    nominal, unit = normalize(nominal)
    # Parse static tolerance
    min_tol_coeff, max_tol_coeff = _normalize_minmax_tuple(tolerance, name="tolerance")
    tol_neg_factor = 1. + min_tol_coeff
    tol_pos_factor = 1. + max_tol_coeff
    # Compute values by tolerance
    tol_min_value = tol_neg_factor * nominal
    tol_max_value = tol_pos_factor * nominal
    # Parse coefficient
    min_coeff, max_coeff = _normalize_minmax_tuple(coefficient, name="coefficient")
    temp_neg_factor = 1. + (tdelta_max * min_coeff) # min_coefficient is typically < 0
    temp_pos_factor = 1. + (tdelta_max * max_coeff)
    # Min: Worst case minimum 
    min_value = temp_neg_factor * tol_min_value
    max_value = temp_pos_factor * tol_max_value

    return ValueRange(
        format_value(min_value, unit, significant_digits=significant_digits),
        format_value(max_value, unit, significant_digits=significant_digits)
    )
    

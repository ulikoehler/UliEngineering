#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing temperature coefficients
and their effects
"""
from collections.abc import Iterable
from typing import Tuple

import numpy as np
from UliEngineering.EngineerIO import NormalizeResult, normalize, normalize_numeric
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args
from UliEngineering.Physics.Temperature import normalize_temperature
from UliEngineering.Utils.Range import normalize_minmax_tuple, ValueRange
from UliEngineering.Electronics.Tolerance import value_range_over_tolerance

__all__ = ["value_range_over_temperature", "value_at_temperature"]

@normalize_numeric_args
def value_at_temperature(nominal, temperature, coefficient="100 ppm", tref="25°C"):
    """
    Given a component with a nominal value (nominal) at a reference temperature (tref)
    and a fixed coefficient of temperature (coefficient, e.g. "100 ppm"),
    computes the actual value of the component at temperature.

    The coefficient of temperature is interpreted in accordance with MIL-STD-202G.
    
    Keyword arguments
    -----------------
    nominal : number or string
        The nominal value of the component e.g. 1023 or "1.023 kΩ"
    coefficient : number or string
        The temperature coefficient of the component per °C
        e.g. "100 ppm", "1 %" or 100e-6
        Note that positive values cause larger values at higher temperatures
        while negative values cause larger values at lower temperatures!
    temperature : number or string
        The temperature to compute the value of the component at
        e.g. "-40 °C". or -40 or "100 K"
        Numbers are interpreted as °C, strings are automatically converted.
    tref : number or string
        The temperature at which the nominal value was measured, i.e. the "reference temperature".
        MIL-STD-202G specifies this as 25°C.
        If not specified in the component datasheet, this is usually "20 °C" or "25 °C".
        Numbers are interpreted as °C, strings are automatically converted.

    Returns
    -------
    A unit-less value representing the value of the component at the given temperature 
    """
    # Note: MIL-STD-202: R-T characteristic: (R2 - R1)/ (R1 * (t2 - t1))
    # Normalize temperatures separately (they need special handling)
    temperature = normalize_temperature(temperature)
    tref = normalize_temperature(tref)
    # Compute (t2 - t1). Might be negative.
    tdelta = temperature - tref
    factor = 1. + (tdelta * coefficient)
    return nominal * factor
    

def value_range_over_temperature(nominal, coefficient:str|float="100ppm", tolerance="0 %", tmin="-40 °C", tmax="85 °C", tref="25 °C", significant_digits=4):
    """
    Given a component which has a nominal value (e.g. "1 kΩ")
    at tref (typically "25 °C") and a coefficient of temperature (e.g. "100ppm").

    Computes the mininimum and maximum possible value of that component
    over the entire temperature range.

    Optionally, a component tolerance can be given (defaults to "0 %")
    to also account for static (temperature-independent) differences.
    Note that the tolerance is applied to the nominal value before
    applying the temperature coefficient.

    The min/max values are computed in accordance with MIL-STD-202 method 304.
    
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
        The nominal value of the component e.g. 1023 or "1.023 kΩ"
    coefficient : number or string or 2-tuple (see above)
        The temperature coefficient of the component per °C
        e.g. "100 ppm", "1 %" or 100e-6 (treated as +/- values)
    tolerance : number or string
        The static (temperature-independent) tolerance of the component.
        e.g. "100 ppm", "1 %" or 100e-6.
        When the value is positive, larger temperatures generate larger values.
        When the value is negative, larger temperatures generate smaller values.
    tmin : number or string
        The minimum temperature to consider in °C, e.g. "-40 °C". or -40 or "100 K"
        Numbers are interpreted as °C, strings are automatically converted.
    tmax : number or string
        The maximum temperature to consider in °C, e.g. "85 °C". or 85 or "300 K".
        Numbers are interpreted as °C, strings are automatically converted.
    tref : number or string
        The temperature at which the nominal value was measured, i.e. the "reference temperature".
        MIL-STD-202G specifies this as 25°C.
        If not specified in the component datasheet, this is usually "20 °C" or "25 °C".
        Numbers are interpreted as °C, strings are automatically converted.
    significant_digits : integer
        How many significant digits to show in the resulting value strings
    Returns
    -------
    A ValueRange() instance containing strings with the correct unit, if any.
    Example: ValueRange("99.5 Ω", "100.5 Ω")
    Use .min and .max to get the min/max value
    """
    # NOTE: These will be in Kelvin after normalization!
    tmin = normalize_temperature(tmin)
    tmax = normalize_temperature(tmax)
    tref = normalize_temperature(tref)
    # Static tolerance
    # We are only interested in the maximum temperature
    # differential from tmin
    tdelta_neg = np.abs(tmin - tref)
    tdelta_pos = np.abs(tmax - tref)
    normalized = normalize(nominal)
    if normalized is None:
        raise ValueError("The nominal value must be a valid number or EngineerIO string.")
    elif isinstance(normalized, Iterable) and not isinstance(normalized, NormalizeResult):
        if len(normalized) == 1 and isinstance(normalized[0], NormalizeResult):
            normalized = normalized[0]
        else:
            raise ValueError("The nominal value must be a single number or EngineerIO string, not a list or array of such objects.")
    nominal, unit = normalized.value, normalized.unit
    # Compute nominal factors by static tolerance
    tol_min_value, tol_max_value, _ = value_range_over_tolerance(nominal, tolerance)
    # Parse coefficient
    tempco = normalize_numeric(coefficient)
    # NOTE: Minimum & maximum value could be any of those (?)
    args = [
        tol_min_value * (1. - (tdelta_neg * tempco)),
        tol_max_value * (1. - (tdelta_neg * tempco)),
        tol_min_value * (1. + (tdelta_pos * tempco)),
        tol_max_value * (1. + (tdelta_pos * tempco))
    ]
    
    min_temp = min(args)
    max_temp = max(args)
 
    return ValueRange(min_temp, max_temp, unit, significant_digits)

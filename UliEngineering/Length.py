#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for length
"""
import scipy.constants
from .EngineerIO import EngineerIO
from .Units import UnknownUnitInContextException, Unit

__all__ = ["normalize_length", "convert_length_to_meters"]

_length_factors = {
    "": 1., # Assumed. It's SI!
    "m": 1.,
    "meter": 1.,
    "meters": 1.,
    'mil': 1e-3 * scipy.constants.inch,
    'in': scipy.constants.inch,
    'inch': scipy.constants.inch,
    'inches': scipy.constants.inch,
    '\"': scipy.constants.inch,
    'foot': scipy.constants.foot,
    'feet': scipy.constants.foot,
    'ft': scipy.constants.foot,
    'yd': scipy.constants.yard,
    'yard': scipy.constants.yard,
    'mile': scipy.constants.mile,
    'miles': scipy.constants.mile,
    'nautical mile': scipy.constants.nautical_mile,
    'nautical miles': scipy.constants.nautical_mile,
    'pt': scipy.constants.point,
    'point': scipy.constants.point,
    'points': scipy.constants.point,
    'AU': scipy.constants.astronomical_unit,
    'au': scipy.constants.astronomical_unit,
    'AUs': scipy.constants.astronomical_unit,
    'ly': scipy.constants.light_year,
    'lightyear': scipy.constants.light_year,
    'lightyears': scipy.constants.light_year,
    'light year': scipy.constants.light_year,
    'light years': scipy.constants.light_year,
    'pc': scipy.constants.parsec,
    'parsec': scipy.constants.parsec,
    'parsecs': scipy.constants.parsec,
    'Å': scipy.constants.angstrom,
    'Angstrom': scipy.constants.angstrom,
    'angstrom': scipy.constants.angstrom,
}

def convert_length_to_meters(value, unit, instance=EngineerIO.length_instance) -> Unit("m"):
    """
    Given a number or Engineer string (unit ignored) <value>
    in <unit>, convert it to meters.
    """
    # Currently a hack, but doing it directly will not parse SI units
    return normalize_length("{} {}".format(value, unit), instance=instance)

def normalize_length(s, instance=EngineerIO.length_instance) -> Unit("m"):
    """
    Normalize a length to meters.
    Returns the numeric value in m or None.

    NOTE: 1 nm is one nanometer, not one nautical mile! Use "1 nautical mile" instead!

    Valid inputs include:
    - "1.0" => 1.0
    - "1.0 mm" => 0.001
    - "1 inch" => 0.0254
    - "1 mil" => 0.000254
    - "1.2 M light years" => 1.135287656709696e+22
    - "9.15 kpc" => 2.8233949868947424e+17
    """
    value, unit = instance.normalize(s)
    if unit in _length_factors:
        return value * _length_factors[unit]
    else:
        raise UnknownUnitInContextException("Unknown length unit: {}".format(unit))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystal oscillator utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit

__all__ = [
    "load_capacitors", "actual_load_capacitance",
    "crystal_deviation_seconds_per_minute",
    "crystal_deviation_seconds_per_hour",
    "crystal_deviation_seconds_per_day",
    "crystal_deviation_seconds_per_month",
    "crystal_deviation_seconds_per_year"
]

def load_capacitors(cload, cpin="3 pF", cstray="2 pF") -> Unit("F"):
    """
    Compute the load capacitors which should be used for a given crystal,
    given that the load capacitors should be symmetric (i.e. have the same value).

    NOTE: You need to use a stray capacitance value that does NOT
    include the parasitic pin capacitance!

    Based on (C1 * C2) / (C1 + C2) + Cstray
    for C1 == C2 == (returned value) + cpin

    >>> auto_format(load_capacitors, "6 pF", cpin="3 pF", cstray="2pF")
    '5.00 pF'

    Parameters
    ----------
    cload : float
        The load capacitance as given in the crystal datasheet
    cstray : float
        The stray capacitance
    cpin : float
        The capacitance of one of the oscillator pins of the connected device
    """
    cload = normalize_numeric(cload)
    cstray = normalize_numeric(cstray)
    cpin = normalize_numeric(cpin)
    # cload = (C1 * C2) / (C1 + C2) + Cstray where C1 == C2
    # => solve A = (B*B) / (B+B) + C for B
    # => solve A = ((B+P)*(B+P)) / ((B+P)+(B+P)) + C for B
    return (2 * (cload - cstray)) - cpin

def actual_load_capacitance(cext, cpin="3 pF", cstray="2 pF") -> Unit("F"):
    """
    Compute the actual load capacitance of a crystal given:

    - The external capacitance value (use "10 pF" if your have a
    10 pF capacitor on each of the crystal pins)
    - The parasitic pin capacitance

    The value returned should match the load capacitance value
    in the crystal datasheet.

    Based on (C1 * C2) / (C1 + C2) + Cstray

    If yu use a

    >>> auto_format(actual_load_capacitance, "5 pF", cpin="3 pF", cstray="2pF")
    '6.00 pF'

    Parameters
    ----------
    cext : float
        The load capacitor value
    cstray : float
        The stray capacitance
    cpin : float
        The capacitance of one of the oscillator pins of the connected device
    """
    cext = normalize_numeric(cext)
    cstray = normalize_numeric(cstray)
    cpin = normalize_numeric(cpin)
    # cload = (C1 * C2) / (C1 + C2) + Cstray where C1 == C2
    # => solve A = (B*B) / (B+B) + C for B
    # => solve A = ((B+P)*(B+P)) / ((B+P)+(B+P)) + C for B
    ctotal = cext + cpin
    return cstray + ((ctotal * ctotal) / (ctotal + ctotal))

def _crystal_deviation_seconds_per_x(deviation, nsecs) -> Unit("s"):
    """Internal common function"""
    deviation = normalize_numeric(deviation) # ppm -> e-6
    return deviation * nsecs

def crystal_deviation_seconds_per_minute(deviation) -> Unit("s"):
    """
    Compute how many seconds a crystal with given ppm
    deviation deviates per hour.

    Use a "n ppm"-like string or use an exponent-(-6)-based number

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_minute, "20 ppm")
    '1.20 ms'
    >>> auto_format(crystal_deviation_seconds_per_minute, 20e-6)
    '1.20 ms'
    """
    return _crystal_deviation_seconds_per_x(deviation, 60)

def crystal_deviation_seconds_per_hour(deviation) -> Unit("s"):
    """
    Compute how many seconds a crystal with given ppm
    deviation deviates per hour.

    Use a "n ppm"-like string or use an exponent-(-6)-based number

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_hour, "20 ppm")
    '72.0 ms'
    >>> auto_format(crystal_deviation_seconds_per_hour, 20e-6)
    '72.0 ms'
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600)

def crystal_deviation_seconds_per_day(deviation) -> Unit("s"):
    """
    Compute how many seconds a crystal with given ppm
    deviation deviates per standard day (24 hours a 3600 seconds)

    Use a "n ppm"-like string or use an exponent-(-6)-based number

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_day, "20 ppm")
    '1.73 s'
    >>> auto_format(crystal_deviation_seconds_per_day, 20e-6)
    '1.73 s'
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600*24)
    
def crystal_deviation_seconds_per_month(deviation) -> Unit("s"):
    """
    Compute how many seconds a crystal with given ppm
    deviation deviates per 31-day month (31 days a 3600*24s)

    Use a "n ppm"-like string or use an exponent-(-6)-based number

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_month, "20 ppm")
    '53.6 s'
    >>> auto_format(crystal_deviation_seconds_per_month, 20e-6)
    '53.6 s'
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600*24*31)

def crystal_deviation_seconds_per_year(deviation) -> Unit("s"):
    """
    Compute how many seconds a crystal with given ppm
    deviation deviates per 365-day year (365 days a 3600*24s)

    Use a "n ppm"-like string or use an exponent-(-6)-based number

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_year, "20 ppm")
    '631 s'
    >>> auto_format(crystal_deviation_seconds_per_year, 20e-6)
    '631 s'
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600*24*365)

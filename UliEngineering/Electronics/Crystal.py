#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crystal oscillator utilities."""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics._normalize import normalize_with_known_units
from .Capacitors import normalize_capacitance, CapacitanceFarad

__all__ = [
    "load_capacitors", "actual_load_capacitance",
    "crystal_deviation_seconds_per_minute",
    "crystal_deviation_seconds_per_hour",
    "crystal_deviation_seconds_per_day",
    "crystal_deviation_seconds_per_month",
    "crystal_deviation_seconds_per_year",
    "normalize_ppm", "PPM",
]


def normalize_ppm(ppm: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(ppm, {"ppm": 1e-6, "ppb": 1e-9, "ppt": 1e-12}, quantity_name="ppm")

PPM = Annotated[NormalizedComputable, normalize_ppm]


@returns_unit("F")
def load_capacitors(cload: CapacitanceFarad, cpin: CapacitanceFarad="3 pF", cstray: CapacitanceFarad="2 pF"):
    """
    Compute the load capacitors which should be used for a given crystal, given that the load capacitors should be symmetric (i.e. have the same value).

    NOTE: You need to use a stray capacitance value that does NOT.
    include the parasitic pin capacitance!

    Based on (C1 * C2) / (C1 + C2) + Cstray
    for C1 == C2 == (returned value) + cpin.

    >>> auto_format(load_capacitors, "6 pF", cpin="3 pF", cstray="2pF")
    '5.00 pF'

    Parameters
    ----------
    cload : CapacitanceFarad
        The load capacitance as given in the crystal datasheet.
    cstray : CapacitanceFarad
        The stray capacitance.
    cpin : CapacitanceFarad
        The capacitance of one of the oscillator pins of the connected device.

    Returns
    -------
    float
        Required load capacitor value in Farads.
    
    """
    # cload = (C1 * C2) / (C1 + C2) + Cstray where C1 == C2
    # => solve A = (B*B) / (B+B) + C for B
    # => solve A = ((B+P)*(B+P)) / ((B+P)+(B+P)) + C for B
    cload = normalize_capacitance(cload) if isinstance(cload, str) else cload
    cpin = normalize_capacitance(cpin) if isinstance(cpin, str) else cpin
    cstray = normalize_capacitance(cstray) if isinstance(cstray, str) else cstray
    return (2 * (cload - cstray)) - cpin

@returns_unit("F")
def actual_load_capacitance(cext: CapacitanceFarad, cpin: CapacitanceFarad="3 pF", cstray: CapacitanceFarad="2 pF"):
    """
    Compute the actual load capacitance of a crystal given.

    - The external capacitance value (use "10 pF" if your have a
    10 pF capacitor on each of the crystal pins).
    - The parasitic pin capacitance.

    The value returned should match the load capacitance value
    in the crystal datasheet.

    Based on (C1 * C2) / (C1 + C2) + Cstray.

    >>> auto_format(actual_load_capacitance, "5 pF", cpin="3 pF", cstray="2pF")
    '6.00 pF'

    Parameters
    ----------
    cext : CapacitanceFarad
        The load capacitor value.
    cstray : CapacitanceFarad
        The stray capacitance.
    cpin : CapacitanceFarad
        The capacitance of one of the oscillator pins of the connected device.

    Returns
    -------
    float
        Actual load capacitance in Farads.
    
    """
    # cload = (C1 * C2) / (C1 + C2) + Cstray where C1 == C2
    # => solve A = (B*B) / (B+B) + C for B
    # => solve A = ((B+P)*(B+P)) / ((B+P)+(B+P)) + C for B
    cext = normalize_capacitance(cext) if isinstance(cext, str) else cext
    cpin = normalize_capacitance(cpin) if isinstance(cpin, str) else cpin
    cstray = normalize_capacitance(cstray) if isinstance(cstray, str) else cstray
    ctotal = cext + cpin
    return cstray + ((ctotal * ctotal) / (ctotal + ctotal))

@returns_unit("s")
def _crystal_deviation_seconds_per_x(deviation: PPM, n_secs):
    """Internal common function."""
    deviation = normalize_ppm(deviation) if isinstance(deviation, str) else deviation
    return deviation * n_secs

@returns_unit("s")
def crystal_deviation_seconds_per_minute(deviation: PPM):
    """
    Compute how many seconds a crystal with given ppm
    
    deviation deviates per minute.

    Use a "n ppm"-like string or use an exponent-(-6)-based number.

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_minute, "20 ppm")
    '1.20 ms'
    >>> auto_format(crystal_deviation_seconds_per_minute, 20e-6)
    '1.20 ms'

    Parameters
    ----------
    deviation : PPM
        Crystal deviation in ppm.

    Returns
    -------
    float
        Deviation in seconds per minute.
    
    """
    return _crystal_deviation_seconds_per_x(deviation, 60)

@returns_unit("s")
def crystal_deviation_seconds_per_hour(deviation: PPM):
    """
    Compute how many seconds a crystal with given ppm
    
    deviation deviates per hour.

    Use a "n ppm"-like string or use an exponent-(-6)-based number.

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_hour, "20 ppm")
    '72.0 ms'
    >>> auto_format(crystal_deviation_seconds_per_hour, 20e-6)
    '72.0 ms'

    Parameters
    ----------
    deviation : PPM
        Crystal deviation in ppm.

    Returns
    -------
    float
        Deviation in seconds per hour.
    
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600)

@returns_unit("s")
def crystal_deviation_seconds_per_day(deviation: PPM):
    """
    Compute how many seconds a crystal with given ppm
    
    deviation deviates per standard day (24 hours a 3600 seconds).

    Use a "n ppm"-like string or use an exponent-(-6)-based number.

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_day, "20 ppm")
    '1.73 s'
    >>> auto_format(crystal_deviation_seconds_per_day, 20e-6)
    '1.73 s'

    Parameters
    ----------
    deviation : PPM
        Crystal deviation in ppm.

    Returns
    -------
    float
        Deviation in seconds per day.
    
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600*24)

@returns_unit("s")
def crystal_deviation_seconds_per_month(deviation: PPM):
    """
    Compute how many seconds a crystal with given ppm
    
    deviation deviates per 31-day month (31 days a 3600*24s).

    Use a "n ppm"-like string or use an exponent-(-6)-based number.

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_month, "20 ppm")
    '53.6 s'
    >>> auto_format(crystal_deviation_seconds_per_month, 20e-6)
    '53.6 s'

    Parameters
    ----------
    deviation : PPM
        Crystal deviation in ppm.

    Returns
    -------
    float
        Deviation in seconds per month.
    
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600*24*31)

@returns_unit("s")
def crystal_deviation_seconds_per_year(deviation: PPM):
    """
    Compute how many seconds a crystal with given ppm
    
    deviation deviates per 365-day year (365 days a 3600*24s).

    Use a "n ppm"-like string or use an exponent-(-6)-based number.

    These calls are equivalent:

    >>> auto_format(crystal_deviation_seconds_per_year, "20 ppm")
    '631 s'
    >>> auto_format(crystal_deviation_seconds_per_year, 20e-6)
    '631 s'

    Parameters
    ----------
    deviation : PPM
        Crystal deviation in ppm.

    Returns
    -------
    float
        Deviation in seconds per year.
    
    """
    return _crystal_deviation_seconds_per_x(deviation, 3600*24*365)

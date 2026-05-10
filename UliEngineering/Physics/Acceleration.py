#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceleration utilities."""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics.Frequency import FrequencyHz
from ._normalize import normalize_with_known_units
import numpy as np
import scipy.constants

g0 = scipy.constants.physical_constants['standard acceleration of gravity'][0]

__all__ = ["g_to_ms2", "ms2_to_g", "centrifugal_acceleration", "centrifuge_radius",
           "normalize_acceleration_ms2", "normalize_acceleration_g",
           "AccelerationMs2", "AccelerationG"]


def normalize_acceleration_ms2(acceleration: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(acceleration, {"m/s²": 1.0, "m/s^2": 1.0, "g": g0, "ms2": 1.0}, quantity_name="acceleration")


def normalize_acceleration_g(acceleration: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(acceleration, {"g": 1.0, "m/s²": 1.0/g0, "m/s^2": 1.0/g0, "ms2": 1.0/g0}, quantity_name="acceleration")


# Unit type annotations
AccelerationMs2 = Annotated[NormalizedComputable, normalize_acceleration_ms2]
AccelerationG = Annotated[NormalizedComputable, normalize_acceleration_g]

@returns_unit("m/s²")
def g_to_ms2(g: AccelerationG):
    """
    Compute the acceleration in m/s² given the acceleration in g.
    """
    g = normalize_acceleration_ms2(g)
    return g * g0

@returns_unit("g")
def ms2_to_g(ms2: AccelerationMs2):
    """
    Compute the acceleration in g given the acceleration in m/s².
    """
    ms2 = normalize_acceleration_ms2(ms2)
    return ms2 / g0

@returns_unit("m/s²")
def centrifugal_acceleration(radius: LengthMeters, speed: FrequencyHz):
    """
    Compute the centrifugal acceleration given

    Online calculator available here:
    https://techoverflow.net/2020/04/20/centrifuge-acceleration-calculator-from-rpm-and-diameter/
    (NOTE: Different units !)

    Parameters
    ----------
    radius :
        The radius of the centrifuge in m
    speed :
        The speed of the centrifuge in Hz

    Returns
    -------
    float
        The acceleration in m/s²
    """
    from UliEngineering.EngineerIO.Length import normalize_length
    from UliEngineering.Physics.Frequency import normalize_frequency
    radius = normalize_length(radius)
    speed = normalize_frequency(speed)
    return 4 * np.pi**2 * radius * speed**2


@returns_unit("m")
def centrifuge_radius(acceleration: AccelerationMs2, speed: FrequencyHz):
    """
    Compute the centrifugal acceleration given

    Online calculator available here:
    https://techoverflow.net/2020/04/20/centrifuge-diameter-calculator-from-acceleration-rpm/
    (NOTE: Different units !)

    Parameters
    ----------
    speed :
        The speed of the centrifuge in Hz
    acceleration:
        The acceleration of the centrifuge in m/s²

    Returns
    -------
    float
        The radius of the centrifuge in m
    """
    from UliEngineering.EngineerIO.Length import normalize_length
    from UliEngineering.Physics.Frequency import normalize_frequency
    acceleration = normalize_acceleration_ms2(acceleration)
    speed = normalize_frequency(speed)
    return acceleration / (4 * np.pi**2 * speed**2)

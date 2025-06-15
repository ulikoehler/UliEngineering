#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for acceleration
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np
import scipy.constants

g0 = scipy.constants.physical_constants['standard acceleration of gravity'][0]

__all__ = ["g_to_ms2", "ms2_to_g", "centrifugal_acceleration", "centrifuge_radius"]

@returns_unit("m/s²")
@normalize_numeric_args
def g_to_ms2(g):
    """
    Compute the acceleration in m/s² given the acceleration in g
    """
    return g * g0

@returns_unit("g")
@normalize_numeric_args
def ms2_to_g(ms2):
    """
    Compute the acceleration in g given the acceleration in m/s²
    """
    return ms2 / g0

@returns_unit("m/s²")
@normalize_numeric_args
def centrifugal_acceleration(radius, speed):
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
    The acceleration in m/s²
    """
    return 4 * np.pi**2 * radius * speed**2


@returns_unit("m")
@normalize_numeric_args
def centrifuge_radius(acceleration, speed):
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
    The radius of the centrifuge in m
    """
    return acceleration / (4 * np.pi**2 * speed**2)

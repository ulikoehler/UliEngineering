#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for acceleration
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np
import scipy.constants

g0 = scipy.constants.physical_constants['standard acceleration of gravity'][0]

__all__ = ["g_to_ms2", "ms2_to_g", "centrifugal_acceleration", "centrifuge_radius"]

def g_to_ms2(g) -> Unit("m/s²"):
    """
    Compute the acceleration in m/s² given the acceleration in g
    """
    g = normalize_numeric(g)
    return g * g0

def ms2_to_g(ms2) -> Unit("g"):
    """
    Compute the acceleration in g given the acceleration in m/s²
    """
    ms2 = normalize_numeric(ms2)
    return ms2 / g0

def centrifugal_acceleration(radius, speed) -> Unit("m/s²"):
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
    radius = normalize_numeric(radius)
    speed = normalize_numeric(speed)
    return 4 * np.pi**2 * radius * speed**2


def centrifuge_radius(acceleration, speed) -> Unit("m"):
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
    acceleration = normalize_numeric(acceleration)
    speed = normalize_numeric(speed)
    return acceleration / (4 * np.pi**2 * speed**2)

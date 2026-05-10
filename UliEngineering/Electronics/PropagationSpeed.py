#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for propagation speed and propagation delay calculations."""
import scipy.constants
import numpy as np

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Length import normalize_length
from UliEngineering.EngineerIO.Types import NormalizableArgument

__all__ = ["propagation_speed", "propagation_delay", "velocity_factor"]


@returns_unit("m/s")
def propagation_speed(e_r: NormalizableArgument = 1.0, mu_r: NormalizableArgument = 1.0):
    """
    Compute the propagation speed in a homogeneous medium characterized
    by the relative permittivity (e_r) and relative permeability (mu_r).

    The formula used is:

    v = c / sqrt(e_r * mu_r).

    Parameters
    ----------
    e_r : float or engineer string
        Relative permittivity (dielectric constant). Default is 1.
    mu_r : float or engineer string
        Relative permeability. Default is 1.

    Returns
    -------
    float
        Propagation speed in m/s.

    Examples
    --------
    >>> propagation_speed(1.0)
    299792458.0
    >>> propagation_speed(4.0)
    149896229.0
    """
    e_r = normalize_numeric(e_r) if isinstance(e_r, str) else e_r
    mu_r = normalize_numeric(mu_r) if isinstance(mu_r, str) else mu_r
    c0 = scipy.constants.c
    return c0 / np.sqrt(e_r * mu_r)


@returns_unit("s")
def propagation_delay(length, e_r: NormalizableArgument = 1.0, mu_r: NormalizableArgument = 1.0):
    """
    Compute the propagation delay for a given physical length in a medium
    with relative permittivity e_r and relative permeability mu_r.

    delay = length / v = length * sqrt(e_r * mu_r) / c.

    Parameters
    ----------
    length : float or engineer string
        Physical length of the transmission line.
    e_r : float or engineer string, optional
        Relative permittivity. Default is 1.
    mu_r : float or engineer string, optional
        Relative permeability. Default is 1.

    Returns
    -------
    float
        Propagation delay in seconds.

    Examples
    --------
    >>> propagation_delay('1 m', 1.0)
    3.3356409519815204e-09
    >>> propagation_delay('1 m', 4.0)
    6.671281903963041e-09
    """
    length = normalize_length(length) if isinstance(length, str) else length
    e_r = normalize_numeric(e_r) if isinstance(e_r, str) else e_r
    mu_r = normalize_numeric(mu_r) if isinstance(mu_r, str) else mu_r
    v = propagation_speed(e_r=e_r, mu_r=mu_r)
    return length / v


@returns_unit("")
def velocity_factor(e_r: NormalizableArgument = 1.0, mu_r: NormalizableArgument = 1.0):
    """
    Return the velocity factor (unitless) for the medium, i.e. the ratio of the
    propagation speed to the speed of light in vacuum.

    velocity_factor = v / c = 1 / sqrt(e_r * mu_r).

    Parameters
    ----------
    e_r : float or engineer string, optional
        Relative permittivity. Default is 1.
    mu_r : float or engineer string, optional
        Relative permeability. Default is 1.

    Returns
    -------
    float
        Velocity factor (unitless).
    """
    e_r = normalize_numeric(e_r) if isinstance(e_r, str) else e_r
    mu_r = normalize_numeric(mu_r) if isinstance(mu_r, str) else mu_r
    return 1.0 / np.sqrt(e_r * mu_r)


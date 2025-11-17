#!/usr/bin/env python3
"""
Utilities for propagation speed and propagation delay calculations
"""
import scipy.constants
import numpy as np

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = ["propagation_speed", "propagation_delay", "velocity_factor"]


@returns_unit("m/s")
@normalize_numeric_args
def propagation_speed(e_r: float = 1.0, mu_r: float = 1.0):
    """
	Compute the propagation speed in a homogeneous medium characterized
	by the relative permittivity (e_r) and relative permeability (mu_r).

	The formula used is:

	v = c / sqrt(e_r * mu_r)

	Parameters
	----------
	e_r : float or engineer string
		Relative permittivity (dielectric constant). Default: 1
	mu_r : float or engineer string
		Relative permeability. Default: 1

	>>> propagation_speed(1.0)
	299792458.0
	>>> propagation_speed(4.0)
	149896229.0
	"""
    c0 = scipy.constants.c
    return c0 / np.sqrt(e_r * mu_r)


@returns_unit("s")
@normalize_numeric_args
def propagation_delay(length, e_r: float = 1.0, mu_r: float = 1.0):
    """
	Compute the propagation delay for a given physical length in a medium
	with relative permittivity e_r and relative permeability mu_r.

	delay = length / v = length * sqrt(e_r * mu_r) / c

	Examples
	--------
	>>> propagation_delay('1 m', 1.0)
	3.3356409519815204e-09
	>>> propagation_delay('1 m', 4.0)
	6.671281903963041e-09
	"""

    v = propagation_speed(e_r=e_r, mu_r=mu_r)
    return length / v


@returns_unit("")
@normalize_numeric_args
def velocity_factor(e_r: float = 1.0, mu_r: float = 1.0):
    """
	Return the velocity factor (unitless) for the medium, i.e. the ratio of the
	propagation speed to the speed of light in vacuum.

	velocity_factor = v / c = 1 / sqrt(e_r * mu_r)
	"""

    return 1.0 / np.sqrt(e_r * mu_r)


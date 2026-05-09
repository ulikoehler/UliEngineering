#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stokes-Einstein relation for diffusion of spherical particles.

The Stokes-Einstein equation relates the diffusion coefficient of a
spherical particle to its radius, solvent viscosity, and temperature:

    D = k_B * T / (6 * π * η * r)

where:
    D   = diffusion coefficient (m²/s)
    k_B = Boltzmann constant (J/K)
    T   = temperature (K)
    η   = dynamic viscosity of the solvent (Pa·s)
    r   = hydrodynamic radius of the particle (m)

Also includes the Stokes-Einstein-Sutherland variant and
the rotational diffusion coefficient.
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from scipy.constants import k as k_B, pi

__all__ = [
    "stokes_einstein_diffusion",
    "stokes_einstein_radius",
    "stokes_einstein_viscosity",
    "stokes_einstein_rotational_diffusion",
    "WATER_VISCOSITY_25C",
]

WATER_VISCOSITY_25C = 8.9e-4  # Pa·s at 25 °C


@normalize_numeric_args
@returns_unit("m²/s")
def stokes_einstein_diffusion(r, eta=WATER_VISCOSITY_25C, T=298.15):
    """
    Compute the translational diffusion coefficient using the
    Stokes-Einstein equation.

    D = k_B * T / (6 * π * η * r)

    Parameters
    ----------
    r : float
        Hydrodynamic radius of the particle in meters.
    eta : float
        Dynamic viscosity of the solvent in Pa·s (default: water at 25 °C).
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Diffusion coefficient in m²/s.
    """
    return k_B * T / (6.0 * pi * eta * r)


@normalize_numeric_args
@returns_unit("m")
def stokes_einstein_radius(D, eta=WATER_VISCOSITY_25C, T=298.15):
    """
    Compute the hydrodynamic radius from diffusion coefficient using
    the inverse Stokes-Einstein equation.

    r = k_B * T / (6 * π * η * D)

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    eta : float
        Dynamic viscosity in Pa·s (default: water at 25 °C).
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Hydrodynamic radius in meters.
    """
    return k_B * T / (6.0 * pi * eta * D)


@normalize_numeric_args
@returns_unit("Pa·s")
def stokes_einstein_viscosity(D, r, T=298.15):
    """
    Compute solvent viscosity from diffusion coefficient and particle radius.

    η = k_B * T / (6 * π * D * r)

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    r : float
        particle radius in meters.
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Dynamic viscosity in Pa·s.
    """
    return k_B * T / (6.0 * pi * D * r)


@normalize_numeric_args
@returns_unit("1/s")
def stokes_einstein_rotational_diffusion(r, eta=WATER_VISCOSITY_25C, T=298.15):
    """
    Compute the rotational diffusion coefficient.

    D_r = k_B * T / (8 * π * η * r³)

    Parameters
    ----------
    r : float
        Hydrodynamic radius in meters.
    eta : float
        Dynamic viscosity in Pa·s (default: water at 25 °C).
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Rotational diffusion coefficient in 1/s.
    """
    return k_B * T / (8.0 * pi * eta * r**3)

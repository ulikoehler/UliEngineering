#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stokes-Einstein relation for diffusion of spherical particles.

The Stokes-Einstein equation relates the diffusion coefficient of a.
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
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature
from scipy.constants import k as k_B, pi

__all__ = [
    "stokes_einstein_diffusion",
    "stokes_einstein_radius",
    "stokes_einstein_viscosity",
    "stokes_einstein_rotational_diffusion",
    "WATER_VISCOSITY_25C",
    "normalize_radius", "RadiusM",
    "normalize_viscosity", "ViscosityPaS",
    "normalize_diffusion_coefficient", "DiffusionCoefficientM2S",
]

WATER_VISCOSITY_25C = 8.9e-4  # Pa·s at 25 °C


def normalize_radius(r: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(r, {"m": 1.0, "nm": 1e-9, "pm": 1e-12, "Å": 1e-10, "cm": 1e-2, "mm": 1e-3, "µm": 1e-6}, quantity_name="radius")

def normalize_viscosity(eta: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(eta, {"Pa·s": 1.0, "Pa*s": 1.0, "cP": 1e-3, "mPa·s": 1e-3, "mPa*s": 1e-3, "P": 0.1}, quantity_name="viscosity")

def normalize_diffusion_coefficient(D: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(D, {"m²/s": 1.0, "m2/s": 1.0, "cm²/s": 1e-4, "cm2/s": 1e-4, "mm²/s": 1e-6, "mm2/s": 1e-6}, quantity_name="diffusion coefficient")

RadiusM = Annotated[NormalizedComputable, normalize_radius]
ViscosityPaS = Annotated[NormalizedComputable, normalize_viscosity]
DiffusionCoefficientM2S = Annotated[NormalizedComputable, normalize_diffusion_coefficient]


@returns_unit("m²/s")
def stokes_einstein_diffusion(r: RadiusM, eta: ViscosityPaS=WATER_VISCOSITY_25C, T=298.15):
    """
    Compute the translational diffusion coefficient using the.
    
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
    r = normalize_radius(r) if isinstance(r, str) else r
    eta = normalize_viscosity(eta) if isinstance(eta, str) else eta
    T = normalize_temperature(T) if isinstance(T, str) else T
    return k_B * T / (6.0 * pi * eta * r)


@returns_unit("m")
def stokes_einstein_radius(D: DiffusionCoefficientM2S, eta: ViscosityPaS=WATER_VISCOSITY_25C, T=298.15):
    """
    Compute the hydrodynamic radius from diffusion coefficient using.
    
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
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    eta = normalize_viscosity(eta) if isinstance(eta, str) else eta
    T = normalize_temperature(T) if isinstance(T, str) else T
    return k_B * T / (6.0 * pi * eta * D)


@returns_unit("Pa·s")
def stokes_einstein_viscosity(D: DiffusionCoefficientM2S, r: RadiusM, T=298.15):
    """
    Compute solvent viscosity from diffusion coefficient and particle radius.

    η = k_B * T / (6 * π * D * r).

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
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    r = normalize_radius(r) if isinstance(r, str) else r
    T = normalize_temperature(T) if isinstance(T, str) else T
    return k_B * T / (6.0 * pi * D * r)


@returns_unit("1/s")
def stokes_einstein_rotational_diffusion(r: RadiusM, eta: ViscosityPaS=WATER_VISCOSITY_25C, T=298.15):
    """
    Compute the rotational diffusion coefficient.

    D_r = k_B * T / (8 * π * η * r³).

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
    r = normalize_radius(r) if isinstance(r, str) else r
    eta = normalize_viscosity(eta) if isinstance(eta, str) else eta
    T = normalize_temperature(T) if isinstance(T, str) else T
    return k_B * T / (8.0 * pi * eta * r**3)

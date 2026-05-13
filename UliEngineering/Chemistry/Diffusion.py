#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fick's laws of diffusion.

Fick's first law (steady-state):
    J = -D * dC/dx

Fick's second law (time-dependent):
    dC/dt = D * d²C/dx²

Also includes diffusion-related utility functions like
mean diffusion distance, diffusion time estimation, and
the error-function concentration profile solution.
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature
import numpy as np
from scipy.special import erfc

__all__ = [
    "fick_first_law",
    "fick_diffusion_distance",
    "fick_diffusion_time",
    "fick_semi_infinite_concentration",
    "fick_thin_film_concentration",
    "diffusion_coefficient_from_temperature",
    "normalize_diffusion_coefficient", "DiffusionCoefficientM2S",
    "normalize_time_seconds", "TimeSeconds",
    "normalize_length", "LengthMeter",
    "normalize_concentration", "ConcentrationMolM3",
    "normalize_energy", "EnergyJPerMol",
]


def normalize_diffusion_coefficient(D: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(D, {"m²/s": 1.0, "m2/s": 1.0, "cm²/s": 1e-4, "cm2/s": 1e-4, "mm²/s": 1e-6, "mm2/s": 1e-6}, quantity_name="diffusion coefficient")

def normalize_time_seconds(t: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(t, {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9, "min": 60.0, "h": 3600.0}, quantity_name="time")

def normalize_length(length: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(length, {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3, "µm": 1e-6, "nm": 1e-9}, quantity_name="length")

def normalize_concentration(conc: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(conc, {"mol/m³": 1.0, "mol/m3": 1.0, "mol/m^3": 1.0, "mol/L": 1000.0, "M": 1000.0, "mM": 1.0, "µM": 1e-3}, quantity_name="concentration")

def normalize_energy(energy: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(energy, {"J/mol": 1.0, "kJ/mol": 1000.0, "eV/mol": 1.602e-19, "cal/mol": 4.184}, quantity_name="energy")

DiffusionCoefficientM2S = Annotated[NormalizedComputable, normalize_diffusion_coefficient]
TimeSeconds = Annotated[NormalizedComputable, normalize_time_seconds]
LengthMeter = Annotated[NormalizedComputable, normalize_length]
ConcentrationMolM3 = Annotated[NormalizedComputable, normalize_concentration]
EnergyJPerMol = Annotated[NormalizedComputable, normalize_energy]


@returns_unit("mol/(m²·s)")
def fick_first_law(D: DiffusionCoefficientM2S, dC_dx):
    """
    Compute the diffusion flux using Fick's first law.

    J = -D * dC/dx.

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    dC_dx : float
        Concentration gradient in mol/m⁴.

    Returns
    -------
    float
        Diffusion flux in mol/(m²·s).

    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    return -D * dC_dx


@returns_unit("m")
def fick_diffusion_distance(D: DiffusionCoefficientM2S, t: TimeSeconds):
    """
    Compute the characteristic (RMS) diffusion distance.

    x_rms = √(2 * D * t).

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    t : float
        Time in seconds.

    Returns
    -------
    float
        RMS diffusion distance in meters.

    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    t = normalize_time_seconds(t) if isinstance(t, str) else t
    return np.sqrt(2.0 * D * t)


@returns_unit("s")
def fick_diffusion_time(D: DiffusionCoefficientM2S, x: LengthMeter):
    """
    Compute the time required for diffusion over a distance x.

    t = x² / (2 * D).

    Parameters
    ----------
    D : float
        Diffusion coefficient in m²/s.
    x : float
        Distance in meters.

    Returns
    -------
    float
        Time in seconds.

    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    x = normalize_length(x) if isinstance(x, str) else x
    return x**2 / (2.0 * D)


@returns_unit("mol/m³")
def fick_semi_infinite_concentration(C0: ConcentrationMolM3, Cs: ConcentrationMolM3, x: LengthMeter, D: DiffusionCoefficientM2S, t: TimeSeconds):
    """
    Compute concentration at distance x and time t using the semi-infinite.
    
    solid solution of Fick's second law (constant surface concentration).

    C(x,t) = Cs - (Cs - C0) * erfc(x / (2*√(D*t)))

    Alternatively: C(x,t) = C0 + (Cs - C0) * erfc(x / (2*√(D*t)))

    Parameters
    ----------
    C0 : float
        Initial uniform concentration in mol/m³.
    Cs : float
        Surface concentration in mol/m³ (constant boundary).
    x : float
        Distance from surface in meters.
    D : float
        Diffusion coefficient in m²/s.
    t : float
        Time in seconds.

    Returns
    -------
    float
        Concentration at position x and time t in mol/m³.

    """
    C0 = normalize_concentration(C0) if isinstance(C0, str) else C0
    Cs = normalize_concentration(Cs) if isinstance(Cs, str) else Cs
    x = normalize_length(x) if isinstance(x, str) else x
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    t = normalize_time_seconds(t) if isinstance(t, str) else t
    return C0 + (Cs - C0) * erfc(x / (2.0 * np.sqrt(D * t)))


@returns_unit("mol/m³")
def fick_thin_film_concentration(M, D: DiffusionCoefficientM2S, t: TimeSeconds, x: LengthMeter):
    """
    Concentration profile from a thin-film (impulse) source diffusing.
    
    in one dimension (Fick's second law, instantaneous plane source).

    C(x,t) = M / √(4πDt) * exp(-x² / (4Dt))

    Parameters
    ----------
    M : float
        Amount of substance per unit area initially deposited (mol/m²).
    D : float
        Diffusion coefficient in m²/s.
    t : float
        Time in seconds.
    x : float
        Distance from the source plane in meters.

    Returns
    -------
    float
        Concentration in mol/m³.

    """
    D = normalize_diffusion_coefficient(D) if isinstance(D, str) else D
    t = normalize_time_seconds(t) if isinstance(t, str) else t
    x = normalize_length(x) if isinstance(x, str) else x
    return M / np.sqrt(4.0 * np.pi * D * t) * np.exp(-x**2 / (4.0 * D * t))


@returns_unit("m²/s")
def diffusion_coefficient_from_temperature(D0: DiffusionCoefficientM2S, Ea: EnergyJPerMol, T):
    """
    Arrhenius-type temperature dependence of diffusion coefficient.

    D(T) = D0 * exp(-Ea / (R * T)).

    Parameters
    ----------
    D0 : float
        Pre-exponential factor in m²/s.
    Ea : float
        Activation energy in J/mol.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Diffusion coefficient at temperature T in m²/s.

    """
    from scipy.constants import R
    D0 = normalize_diffusion_coefficient(D0) if isinstance(D0, str) else D0
    Ea = normalize_energy(Ea) if isinstance(Ea, str) else Ea
    T = normalize_temperature(T) if isinstance(T, str) else T
    return D0 * np.exp(-Ea / (R * T))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive viscosity module.

Provides functions and dataclass constants for:
- Andrade viscosity (Arrhenius-type temperature dependence)
- VFT viscosity (Vogel-Fulcher-Tammann)
- Sutherland gas viscosity
- Swindells water viscosity correlation
- Kestin viscosity correlation
- Bingham plastic model
- Poiseuille's law for laminar pipe flow
- Kinematic viscosity
- Stokes' law for drag on a sphere
- Reynolds number
- Arrhenius mixing rule for mixture viscosity

All temperatures are in Kelvin unless otherwise noted.
Dynamic viscosities are returned in Pa·s (Pa s).
Kinematic viscosities are returned in m²/s.

"""

import numpy as np
from dataclasses import dataclass
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics.Temperature import TemperatureKelvin, normalize_temperature
from ._normalize import normalize_with_known_units

__all__ = [
    # Dataclasses
    "AndradeConstants",
    "VFTConstants",
    "SutherlandConstants",
    "SwindellsConstants",
    "KestinConstants",
    "BinghamConstants",
    "LiquidViscosityData",
    "GasViscosityData",
    # Functions
    "andrade_viscosity",
    "vft_viscosity",
    "sutherland_gas_viscosity",
    "swindells_viscosity",
    "kestin_viscosity",
    "bingham_stress",
    "poiseuille_flow_rate",
    "kinematic_viscosity",
    "stokes_drag",
    "reynolds_number",
    "arrhenius_mixing_viscosity",
    # Pre-defined constants
    "CommonLiquids",
    "CommonGases",
    # Normalize functions and types
    "normalize_dynamic_viscosity", "DynamicViscosityPas",
    "normalize_density", "DensityKgM3",
    "normalize_length", "LengthMeter",
    "normalize_pressure", "PressurePascal",
    "normalize_velocity", "VelocityMS",
    "normalize_shear_rate", "ShearRate",
    "normalize_temperature", "TemperatureKelvin",
]


# ---------------------------------------------------------------------------
# Normalize functions and Annotated types
# ---------------------------------------------------------------------------

def normalize_dynamic_viscosity(viscosity: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(viscosity, {"Pa·s": 1.0, "Pa s": 1.0, "Pas": 1.0, "mPa·s": 1e-3, "cP": 1e-3, "P": 0.1}, quantity_name="dynamic viscosity")

def normalize_density(density: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(density, {"kg/m³": 1.0, "kg/m3": 1.0, "kg/m^3": 1.0, "g/cm³": 1000.0, "g/cm3": 1000.0, "g/L": 1.0}, quantity_name="density")

def normalize_length(length: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(length, {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3, "µm": 1e-6, "nm": 1e-9}, quantity_name="length")

def normalize_pressure(pressure: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(pressure, {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "bar": 1e5, "mbar": 100}, quantity_name="pressure")

def normalize_velocity(velocity: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(velocity, {"m/s": 1.0, "m/s²": 1.0, "km/h": 0.2777777777777778, "mph": 0.44704}, quantity_name="velocity")

def normalize_shear_rate(shear_rate: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(shear_rate, {"s⁻¹": 1.0, "/s": 1.0}, quantity_name="shear rate")

DynamicViscosityPas = Annotated[NormalizedComputable, normalize_dynamic_viscosity]
DensityKgM3 = Annotated[NormalizedComputable, normalize_density]
LengthMeter = Annotated[NormalizedComputable, normalize_length]
PressurePascal = Annotated[NormalizedComputable, normalize_pressure]
VelocityMS = Annotated[NormalizedComputable, normalize_velocity]
ShearRate = Annotated[NormalizedComputable, normalize_shear_rate]


# ---------------------------------------------------------------------------
# Dataclasses for material-specific constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AndradeConstants:
    """
    Constants for the Andrade (Arrhenius-type) viscosity equation:

        η = A * exp(B / T)

    Parameters
    ----------
    name : str
        Name of the fluid.
    A : float
        Pre-exponential factor (Pa·s).
    B : float
        Activation-energy-like parameter (K).
    """
    name: str
    A: float
    B: float


@dataclass(frozen=True)
class VFTConstants:
    """
    Constants for the Vogel-Fulcher-Tammann (VFT) viscosity equation:

        η = A * exp(B / (T - T0))

    Parameters
    ----------
    name : str
        Name of the fluid.
    A : float
        Pre-exponential factor (Pa·s).
    B : float
        Fragility parameter (K).
    T0 : float
        Vogel temperature (K).  Must be strictly less than the operating T.
    """
    name: str
    A: float
    B: float
    T0: float


@dataclass(frozen=True)
class SutherlandConstants:
    """
    Constants for the Sutherland gas-viscosity model:

        μ = μ0 * (T0 + C) / (T + C) * (T / T0)^(3/2)

    Parameters
    ----------
    name : str
        Name of the gas.
    mu0 : float
        Reference dynamic viscosity (Pa·s).
    T0 : float
        Reference temperature (K).
    C : float
        Sutherland constant (K).
    """
    name: str
    mu0: float
    T0: float
    C: float


@dataclass(frozen=True)
class SwindellsConstants:
    """
    Constants for the Swindells empirical viscosity correlation:

        η = η_ref * 10^( -a * (T - T_ref) / (T + b) )

    Parameters
    ----------
    name : str
        Name of the fluid.
    eta_ref : float
        Reference dynamic viscosity (Pa·s).
    T_ref : float
        Reference temperature (K).
    a : float
        Dimensionless slope parameter.
    b : float
        Temperature offset (K).
    """
    name: str
    eta_ref: float
    T_ref: float
    a: float
    b: float


@dataclass(frozen=True)
class KestinConstants:
    """
    Constants for the Kestin (three-parameter) viscosity correlation:

        η = A * exp( B / (T - C) )

    Parameters
    ----------
    name : str
        Name of the fluid.
    A : float
        Pre-exponential factor (Pa·s).
    B : float
        Energy-like parameter (K).
    C : float
        Temperature offset (K).  Must be strictly less than operating T.
    """
    name: str
    A: float
    B: float
    C: float


@dataclass(frozen=True)
class BinghamConstants:
    """
    Constants for a Bingham plastic fluid:

        τ = τ0 + μ_p * γ̇

    Parameters
    ----------
    name : str
        Name of the material.
    tau0 : float
        Yield stress (Pa).
    mu_p : float
        Plastic viscosity (Pa·s).
    """
    name: str
    tau0: float
    mu_p: float


@dataclass(frozen=True)
class LiquidViscosityData:
    """
    Bundle of viscosity-model constants for a single liquid.

    Parameters
    ----------
    name : str
        Liquid name.
    andrade : AndradeConstants, optional
    vft : VFTConstants, optional
    swindells : SwindellsConstants, optional
    kestin : KestinConstants, optional
    density_kg_m3 : float, optional
        Typical density at room temperature (kg/m³).
    """
    name: str
    andrade: AndradeConstants = None
    vft: VFTConstants = None
    swindells: SwindellsConstants = None
    kestin: KestinConstants = None
    density_kg_m3: float = None


@dataclass(frozen=True)
class GasViscosityData:
    """
    Bundle of viscosity-model constants for a single gas.

    Parameters
    ----------
    name : str
        Gas name.
    sutherland : SutherlandConstants
    density_kg_m3 : float, optional
        Typical density at STP (kg/m³).
    """
    name: str
    sutherland: SutherlandConstants = None
    density_kg_m3: float = None


# ---------------------------------------------------------------------------
# Pre-defined constants for common fluids
# ---------------------------------------------------------------------------

class CommonLiquids:
    """Pre-defined viscosity constants for common liquids."""

    Water = LiquidViscosityData(
        name="Water",
        andrade=AndradeConstants(name="Water", A=1.80e-6, B=1885.0),
        vft=VFTConstants(name="Water", A=3.05e-5, B=489.3, T0=153.0),
        swindells=SwindellsConstants(
            name="Water", eta_ref=1.002e-3, T_ref=293.15, a=1.524, b=-152.0
        ),
        kestin=KestinConstants(name="Water", A=2.414e-5, B=247.8, C=140.0),
        density_kg_m3=998.2,
    )

    Ethanol = LiquidViscosityData(
        name="Ethanol",
        andrade=AndradeConstants(name="Ethanol", A=1.0e-7, B=2000.0),
        vft=VFTConstants(name="Ethanol", A=5.0e-6, B=600.0, T0=120.0),
        density_kg_m3=789.0,
    )

    Methanol = LiquidViscosityData(
        name="Methanol",
        andrade=AndradeConstants(name="Methanol", A=1.5e-7, B=1800.0),
        vft=VFTConstants(name="Methanol", A=3.0e-6, B=500.0, T0=110.0),
        density_kg_m3=791.8,
    )

    Glycerol = LiquidViscosityData(
        name="Glycerol",
        andrade=AndradeConstants(name="Glycerol", A=1.0e-10, B=5000.0),
        vft=VFTConstants(name="Glycerol", A=1.0e-8, B=1500.0, T0=180.0),
        density_kg_m3=1261.0,
    )

    OliveOil = LiquidViscosityData(
        name="Olive Oil",
        andrade=AndradeConstants(name="Olive Oil", A=5.0e-5, B=2500.0),
        vft=VFTConstants(name="Olive Oil", A=1.0e-4, B=800.0, T0=200.0),
        density_kg_m3=920.0,
    )

    Mercury = LiquidViscosityData(
        name="Mercury",
        andrade=AndradeConstants(name="Mercury", A=7.0e-4, B=500.0),
        vft=VFTConstants(name="Mercury", A=1.0e-3, B=200.0, T0=50.0),
        density_kg_m3=13534.0,
    )

    Acetone = LiquidViscosityData(
        name="Acetone",
        andrade=AndradeConstants(name="Acetone", A=2.0e-7, B=1500.0),
        vft=VFTConstants(name="Acetone", A=1.0e-6, B=400.0, T0=100.0),
        density_kg_m3=784.0,
    )

    Benzene = LiquidViscosityData(
        name="Benzene",
        andrade=AndradeConstants(name="Benzene", A=3.0e-7, B=1600.0),
        vft=VFTConstants(name="Benzene", A=2.0e-6, B=450.0, T0=90.0),
        density_kg_m3=876.5,
    )


class CommonGases:
    """Pre-defined Sutherland constants for common gases."""

    Air = GasViscosityData(
        name="Air",
        sutherland=SutherlandConstants(name="Air", mu0=1.716e-5, T0=273.15, C=111.0),
        density_kg_m3=1.225,
    )

    Nitrogen = GasViscosityData(
        name="Nitrogen",
        sutherland=SutherlandConstants(name="Nitrogen", mu0=1.663e-5, T0=273.15, C=107.0),
        density_kg_m3=1.145,
    )

    Oxygen = GasViscosityData(
        name="Oxygen",
        sutherland=SutherlandConstants(name="Oxygen", mu0=1.919e-5, T0=273.15, C=139.0),
        density_kg_m3=1.308,
    )

    CarbonDioxide = GasViscosityData(
        name="Carbon Dioxide",
        sutherland=SutherlandConstants(
            name="Carbon Dioxide", mu0=1.370e-5, T0=273.15, C=222.0
        ),
        density_kg_m3=1.842,
    )

    Helium = GasViscosityData(
        name="Helium",
        sutherland=SutherlandConstants(name="Helium", mu0=1.865e-5, T0=273.15, C=79.4),
        density_kg_m3=0.164,
    )


# ---------------------------------------------------------------------------
# Core viscosity functions
# ---------------------------------------------------------------------------

def _to_array(x):
    """Helper to convert scalar or iterable to numpy array."""
    if isinstance(x, (np.ndarray, list, tuple)):
        return np.asarray(x, dtype=float)
    return x


@returns_unit("Pa·s")
def andrade_viscosity(T: TemperatureKelvin, constants: AndradeConstants = CommonLiquids.Water.andrade):
    """
    Compute dynamic viscosity using the Andrade (Arrhenius-type) equation.

        η = A * exp(B / T)

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.  Must be strictly positive.
    constants : AndradeConstants
        Material constants (default: Water).

    Returns
    -------
    float or ndarray
        Dynamic viscosity in Pa·s.

    Raises
    ------
    ValueError
        If T <= 0.
    """
    T = normalize_temperature(T) if isinstance(T, str) else T
    T = _to_array(T)
    if np.any(T <= 0):
        raise ValueError("Temperature T must be strictly positive for Andrade viscosity.")
    return constants.A * np.exp(constants.B / T)


@returns_unit("Pa·s")
def vft_viscosity(T: TemperatureKelvin, constants: VFTConstants = CommonLiquids.Water.vft):
    """
    Compute dynamic viscosity using the Vogel-Fulcher-Tammann (VFT) equation.

        η = A * exp( B / (T - T0) )

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.  Must be strictly greater than T0.
    constants : VFTConstants
        Material constants (default: Water).

    Returns
    -------
    float or ndarray
        Dynamic viscosity in Pa·s.

    Raises
    ------
    ValueError
        If T <= T0.
    """
    T = normalize_temperature(T) if isinstance(T, str) else T
    T = _to_array(T)
    if np.any(T <= constants.T0):
        raise ValueError(
            f"Temperature T must be strictly greater than T0 ({constants.T0} K) "
            "for VFT viscosity."
        )
    return constants.A * np.exp(constants.B / (T - constants.T0))


@returns_unit("Pa·s")
def sutherland_gas_viscosity(
    T: TemperatureKelvin, constants: SutherlandConstants = CommonGases.Air.sutherland
):
    """
    Compute dynamic viscosity of a gas using the Sutherland model.

        μ = μ0 * (T0 + C) / (T + C) * (T / T0)^(3/2)

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.  Must be strictly positive.
    constants : SutherlandConstants
        Material constants (default: Air).

    Returns
    -------
    float or ndarray
        Dynamic viscosity in Pa·s.

    Raises
    ------
    ValueError
        If T <= 0.
    """
    T = normalize_temperature(T) if isinstance(T, str) else T
    T = _to_array(T)
    if np.any(T <= 0):
        raise ValueError(
            "Temperature T must be strictly positive for Sutherland gas viscosity."
        )
    T0 = constants.T0
    C = constants.C
    mu0 = constants.mu0
    return mu0 * (T0 + C) / (T + C) * (T / T0) ** 1.5


@returns_unit("Pa·s")
def swindells_viscosity(
    T: TemperatureKelvin, constants: SwindellsConstants = CommonLiquids.Water.swindells
):
    """
    Compute dynamic viscosity using the Swindells empirical correlation.

        η = η_ref * 10^( -a * (T - T_ref) / (T + b) )

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.
    constants : SwindellsConstants
        Material constants (default: Water).

    Returns
    -------
    float or ndarray
        Dynamic viscosity in Pa·s.
    """
    T = normalize_temperature(T) if isinstance(T, str) else T
    T = _to_array(T)
    exponent = -constants.a * (T - constants.T_ref) / (T + constants.b)
    return constants.eta_ref * np.power(10.0, exponent)


@returns_unit("Pa·s")
def kestin_viscosity(T: TemperatureKelvin, constants: KestinConstants = CommonLiquids.Water.kestin):
    """
    Compute dynamic viscosity using the Kestin three-parameter correlation.

        η = A * exp( B / (T - C) )

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.  Must be strictly greater than C.
    constants : KestinConstants
        Material constants (default: Water).

    Returns
    -------
    float or ndarray
        Dynamic viscosity in Pa·s.

    Raises
    ------
    ValueError
        If T <= C.
    """
    T = normalize_temperature(T) if isinstance(T, str) else T
    T = _to_array(T)
    if np.any(T <= constants.C):
        raise ValueError(
            f"Temperature T must be strictly greater than C ({constants.C} K) "
            "for Kestin viscosity."
        )
    return constants.A * np.exp(constants.B / (T - constants.C))


@returns_unit("Pa")
def bingham_stress(shear_rate: ShearRate, constants: BinghamConstants = None):
    """
    Compute shear stress for a Bingham plastic fluid.

        τ = τ0 + μ_p * γ̇

    Parameters
    ----------
    shear_rate : float or array-like
        Shear rate γ̇ in s⁻¹.  Must be non-negative.
    constants : BinghamConstants
        Material constants.  If None, a default example (τ0 = 10 Pa, μ_p = 0.1 Pa·s)
        is used.

    Returns
    -------
    float or ndarray
        Shear stress in Pa.

    Raises
    ------
    ValueError
        If shear_rate < 0.
    """
    if constants is None:
        constants = BinghamConstants(name="Example Bingham fluid", tau0=10.0, mu_p=0.1)
    gamma = normalize_shear_rate(shear_rate) if isinstance(shear_rate, str) else shear_rate
    gamma = _to_array(gamma)
    if np.any(gamma < 0):
        raise ValueError("Shear rate must be non-negative for Bingham model.")
    return constants.tau0 + constants.mu_p * gamma


@returns_unit("m³/s")
def poiseuille_flow_rate(radius: LengthMeter, pressure_drop: PressurePascal, length: LengthMeter, viscosity: DynamicViscosityPas):
    """
    Compute volumetric flow rate for laminar flow in a cylindrical pipe using Poiseuille's law.

        Q = π * r⁴ * ΔP / (8 * η * L)

    Parameters
    ----------
    radius : float or array-like
        Pipe radius in m.  Must be strictly positive.
    pressure_drop : float or array-like
        Pressure difference ΔP along the pipe in Pa.
    length : float or array-like
        Pipe length in m.  Must be strictly positive.
    viscosity : float or array-like
        Dynamic viscosity η in Pa·s.  Must be strictly positive.

    Returns
    -------
    float or ndarray
        Volumetric flow rate in m³/s.

    Raises
    ------
    ValueError
        If radius, length, or viscosity is not strictly positive.
    """
    r = normalize_length(radius) if isinstance(radius, str) else radius
    L = normalize_length(length) if isinstance(length, str) else length
    eta = normalize_dynamic_viscosity(viscosity) if isinstance(viscosity, str) else viscosity
    dP = normalize_pressure(pressure_drop) if isinstance(pressure_drop, str) else pressure_drop
    r = _to_array(r)
    L = _to_array(L)
    eta = _to_array(eta)
    dP = _to_array(dP)
    if np.any(r <= 0):
        raise ValueError("Pipe radius must be strictly positive.")
    if np.any(L <= 0):
        raise ValueError("Pipe length must be strictly positive.")
    if np.any(eta <= 0):
        raise ValueError("Viscosity must be strictly positive.")
    return np.pi * r ** 4 * dP / (8.0 * eta * L)


@returns_unit("m²/s")
def kinematic_viscosity(dynamic_viscosity: DynamicViscosityPas, density: DensityKgM3):
    """
    Compute kinematic viscosity from dynamic viscosity and density.

        ν = η / ρ

    Parameters
    ----------
    dynamic_viscosity : float or array-like
        Dynamic viscosity η in Pa·s.
    density : float or array-like
        Density ρ in kg/m³.  Must be strictly positive.

    Returns
    -------
    float or ndarray
        Kinematic viscosity in m²/s.

    Raises
    ------
    ValueError
        If density is not strictly positive.
    """
    eta = normalize_dynamic_viscosity(dynamic_viscosity) if isinstance(dynamic_viscosity, str) else dynamic_viscosity
    rho = normalize_density(density) if isinstance(density, str) else density
    eta = _to_array(eta)
    rho = _to_array(rho)
    if np.any(rho <= 0):
        raise ValueError("Density must be strictly positive.")
    return eta / rho


@returns_unit("N")
def stokes_drag(radius: LengthMeter, velocity: VelocityMS, viscosity: DynamicViscosityPas):
    """
    Compute drag force on a sphere moving in a viscous fluid (Stokes' law).

        F = 6 * π * η * r * v

    Parameters
    ----------
    radius : float or array-like
        Sphere radius in m.  Must be non-negative.
    velocity : float or array-like
        Velocity in m/s.
    viscosity : float or array-like
        Dynamic viscosity η in Pa·s.  Must be non-negative.

    Returns
    -------
    float or ndarray
        Drag force in N.

    Raises
    ------
    ValueError
        If radius or viscosity is negative.
    """
    r = normalize_length(radius) if isinstance(radius, str) else radius
    v = normalize_velocity(velocity) if isinstance(velocity, str) else velocity
    eta = normalize_dynamic_viscosity(viscosity) if isinstance(viscosity, str) else viscosity
    r = _to_array(r)
    v = _to_array(v)
    eta = _to_array(eta)
    if np.any(r < 0):
        raise ValueError("Sphere radius must be non-negative.")
    if np.any(eta < 0):
        raise ValueError("Viscosity must be non-negative.")
    return 6.0 * np.pi * eta * r * v


@returns_unit("")
def reynolds_number(density: DensityKgM3, velocity: VelocityMS, characteristic_length: LengthMeter, viscosity: DynamicViscosityPas):
    """
    Compute the Reynolds number.

        Re = ρ * v * L / η

    Parameters
    ----------
    density : float or array-like
        Density ρ in kg/m³.  Must be strictly positive.
    velocity : float or array-like
        Velocity v in m/s.
    characteristic_length : float or array-like
        Characteristic length L in m (e.g. pipe diameter).  Must be strictly positive.
    viscosity : float or array-like
        Dynamic viscosity η in Pa·s.  Must be strictly positive.

    Returns
    -------
    float or ndarray
        Dimensionless Reynolds number.

    Raises
    ------
    ValueError
        If density, characteristic_length, or viscosity is not strictly positive.
    """
    rho = normalize_density(density) if isinstance(density, str) else density
    v = normalize_velocity(velocity) if isinstance(velocity, str) else velocity
    L = normalize_length(characteristic_length) if isinstance(characteristic_length, str) else characteristic_length
    eta = normalize_dynamic_viscosity(viscosity) if isinstance(viscosity, str) else viscosity
    rho = _to_array(rho)
    v = _to_array(v)
    L = _to_array(L)
    eta = _to_array(eta)
    if np.any(rho <= 0):
        raise ValueError("Density must be strictly positive for Reynolds number.")
    if np.any(L <= 0):
        raise ValueError("Characteristic length must be strictly positive.")
    if np.any(eta <= 0):
        raise ValueError("Viscosity must be strictly positive for Reynolds number.")
    return rho * v * L / eta


@returns_unit("Pa·s")
def arrhenius_mixing_viscosity(mole_fractions, viscosities):
    """
    Compute mixture viscosity using the Arrhenius (logarithmic) mixing rule.

        ln(η_mix) = Σ xi * ln(ηi)

    Parameters
    ----------
    mole_fractions : iterable of float
        Mole fractions xi.  Must sum to 1.0 (within tolerance).
    viscosities : iterable of float
        Component dynamic viscosities ηi in Pa·s.  Each must be strictly positive.

    Returns
    -------
    float
        Mixture dynamic viscosity in Pa·s.

    Raises
    ------
    ValueError
        If mole fractions do not sum to 1.0, or if any viscosity is not positive,
        or if inputs have mismatched lengths.
    """
    xi = np.asarray(mole_fractions, dtype=float)
    eta_i = np.asarray(viscosities, dtype=float)
    if xi.shape != eta_i.shape:
        raise ValueError(
            "mole_fractions and viscosities must have the same length."
        )
    if len(xi) == 0:
        raise ValueError("Input arrays must not be empty.")
    if np.any(eta_i <= 0):
        raise ValueError("All component viscosities must be strictly positive.")
    if np.any(xi < 0):
        raise ValueError("Mole fractions must be non-negative.")
    if not np.isclose(np.sum(xi), 1.0, atol=1e-6):
        raise ValueError(f"Mole fractions must sum to 1.0 (got {np.sum(xi)}).")
    return np.exp(np.sum(xi * np.log(eta_i)))

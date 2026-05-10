#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pressure utilities."""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Length import LengthMeters
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.EngineerIO import normalize_numeric
from ._normalize import normalize_with_known_units

__all__ = ["pascal_to_bar", "bar_to_pascal", "barlow_tangential",
           "psi_to_pascal", "psi_to_bar", "pascal_to_psi", "bar_to_psi",
           "normalize_pressure_pascal", "normalize_pressure_bar",
           "PressurePascal", "PressureBar"]


def normalize_pressure_pascal(pressure: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(pressure, {"Pa": 1.0, "bar": 1e5, "psi": 6894.76}, quantity_name="pressure")


def normalize_pressure_bar(pressure: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(pressure, {"bar": 1.0, "Pa": 1e-5, "psi": 0.0689476}, quantity_name="pressure")


# Unit type annotations
PressurePascal = Annotated[NormalizedComputable, normalize_pressure_pascal]
PressureBar = Annotated[NormalizedComputable, normalize_pressure_bar]

@returns_unit("bar")
def pascal_to_bar(pressure: PressurePascal):
    """
    Convert the pressure in pascal to the pressure in bar
    """
    pressure = normalize_pressure_pascal(pressure) if isinstance(pressure, str) else pressure
    return pressure*1e-5

@returns_unit("Pa")
def bar_to_pascal(pressure: PressureBar):
    """
    Convert the pressure in bar to the pressure in Pascal
    """
    pressure = normalize_pressure_bar(pressure) if isinstance(pressure, str) else pressure
    return pressure*1e5

@returns_unit("Pa")
def barlow_tangential(outer_diameter: LengthMeters, inner_diameter: LengthMeters, pressure: PressurePascal):
    """
    Compute the tangential stress of a pressure vessel at [pressure] using Barlow's formula for thin-walled tubes.

    Note that this formula only applies for (outer_diameter/inner_diameter) < 1.2 !
    (this assumption is not checked). Otherwise, the stress distribution will be too uneven.
    """
    outer_diameter = normalize_numeric(outer_diameter) if isinstance(outer_diameter, str) else outer_diameter
    inner_diameter = normalize_numeric(inner_diameter) if isinstance(inner_diameter, str) else inner_diameter
    pressure = normalize_pressure_pascal(pressure) if isinstance(pressure, str) else pressure
    dm = (outer_diameter + inner_diameter) / 2
    s = (outer_diameter - inner_diameter) / 2
    return pressure * dm / (2 * s)


@returns_unit("Pa")
def psi_to_pascal(pressure: PressurePascal):
    """
    Convert the pressure in psi to the pressure in Pascal
    """
    pressure = normalize_pressure_pascal(pressure) if isinstance(pressure, str) else pressure
    return pressure * 6894.76


@returns_unit("bar")
def psi_to_bar(pressure: PressurePascal):
    """
    Convert the pressure in psi to the pressure in bar
    """
    pressure = normalize_pressure_pascal(pressure) if isinstance(pressure, str) else pressure
    return pressure * 0.0689476


@returns_unit("psi")
def pascal_to_psi(pressure: PressurePascal):
    """
    Convert the pressure in Pascal to the pressure in psi
    """
    pressure = normalize_pressure_pascal(pressure) if isinstance(pressure, str) else pressure
    return pressure / 6894.76


@returns_unit("psi")
def bar_to_psi(pressure: PressureBar):
    """
    Convert the pressure in bar to the pressure in psi
    """
    pressure = normalize_pressure_bar(pressure) if isinstance(pressure, str) else pressure
    return pressure / 0.0689476

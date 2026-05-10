#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for acceleration
"""
from typing import Annotated

from UliEngineering.EngineerIO import EngineerIO
from UliEngineering.EngineerIO.Decorators import normalize_args, returns_unit
from UliEngineering.EngineerIO.Types import NormalizedComputable
from UliEngineering.Units import Hz, InvalidUnitInContextException, m
import numpy as np

__all__ = ["rpm_to_Hz", "rpm_to_rps", "hz_to_rpm", "angular_speed",
           "rotation_linear_speed", "centrifugal_force", "normalize_frequency"]


def _normalize_with_known_units(value, unit_factors, default_factor=1.0, quantity_name="value"):
    if value is None:
        raise ValueError(f"Can't normalize {quantity_name} None")
    if isinstance(value, list):
        return [_normalize_with_known_units(item, unit_factors, default_factor, quantity_name) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_with_known_units(item, unit_factors, default_factor, quantity_name) for item in value)
    if isinstance(value, np.ndarray):
        return np.asarray([_normalize_with_known_units(item, unit_factors, default_factor, quantity_name) for item in value])
    if isinstance(value, (int, float, np.generic)):
        return float(value) * default_factor

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported {quantity_name} value type: {type(value)!r}")

    raw_value = value.strip()
    compact_value = value.strip().replace(" ", "")
    for unit, factor in sorted(unit_factors.items(), key=lambda item: len(item[0]), reverse=True):
        if compact_value.endswith(unit):
            numeric_part = compact_value[:-len(unit)]
            if not numeric_part:
                raise ValueError(f"Missing numeric part in {quantity_name} string '{value}'")
            return EngineerIO.instance().normalize_numeric(numeric_part) * factor
    if any(ch.isspace() for ch in raw_value) or "/" in compact_value or "^" in compact_value:
        raise InvalidUnitInContextException(
            f"Invalid unit in {quantity_name} string '{value}'. Expected one of: {', '.join(sorted(unit_factors))}"
        )
    return EngineerIO.instance().normalize_numeric(compact_value) * default_factor


def normalize_rpm(speed):
    return _normalize_with_known_units(speed, {"rpm": 1.0}, quantity_name="rotational speed")


def normalize_frequency(speed):
    return _normalize_with_known_units(speed, {"rpm": 1.0 / 60.0, "Hz": 1.0}, quantity_name="frequency")


def normalize_mass_grams(mass):
    return _normalize_with_known_units(mass, {"kg": 1000.0, "mg": 0.001, "g": 1.0}, quantity_name="mass")


def normalize_density_kg_per_m3(density):
    return _normalize_with_known_units(
        density,
        {
            "kg/m^3": 1.0,
            "kg/m3": 1.0,
            "g/cm^3": 1000.0,
            "g/cm3": 1000.0,
            "g/L": 1.0,
            "g/l": 1.0,
        },
        quantity_name="density",
    )


RotationFrequency = Annotated[NormalizedComputable, normalize_frequency]
RotationRateRpm = Annotated[NormalizedComputable, normalize_rpm]
LengthMeters = Annotated[NormalizedComputable, m]
MassGrams = Annotated[NormalizedComputable, normalize_mass_grams]
DensityKgPerM3 = Annotated[NormalizedComputable, normalize_density_kg_per_m3]

@returns_unit("Hz")
@normalize_args
def rpm_to_Hz(rpm_value: RotationRateRpm) -> NormalizedComputable:
    """
    Compute the rotational speed in Hz given the rotational speed in rpm
    """
    return rpm_value / 60.

@returns_unit("rpm")
@normalize_args
def hz_to_rpm(speed: Annotated[NormalizedComputable, Hz]) -> NormalizedComputable:
    """
    Compute the rotational speed in rpm given the rotational speed in Hz
    """
    return speed * 60.

rpm_to_rps = rpm_to_Hz

@returns_unit("1/s")
@normalize_args
def angular_speed(speed: RotationFrequency) -> NormalizedComputable:
    """
    Compute Ω, the angular speed of a centrifugal system
    """
    return 2*np.pi*speed

@returns_unit("m/s")
@normalize_args
def rotation_linear_speed(radius: LengthMeters, speed: RotationFrequency) -> NormalizedComputable:
    """
    Compute the linear speed at a given [radius] for a centrifugal system rotating at [speed].
    """
    return radius * angular_speed(speed)

@returns_unit("N")
@normalize_args
def centrifugal_force(radius: LengthMeters, speed: RotationFrequency, mass: MassGrams) -> NormalizedComputable:
    """
    Compute the centrifugal force of a [mass] rotation at [speed] at radius [radius]
    """
    mass = mass / 1000.0 # mass needs to be Kilograms TODO Improve
    return mass * angular_speed(speed)**2 * radius

@returns_unit("Pa")
@normalize_args
def rotating_liquid_pressure(density: DensityKgPerM3, speed: RotationFrequency, radius: LengthMeters) -> NormalizedComputable:
    """
    Compute the pressure in a body of liquid (relative to the steady-state pressure)
    The calculation does not include gravity.

    Also see https://www.youtube.com/watch?v=kIH7wEq3H-M
    Also see https://www.physicsforums.com/threads/pressure-of-a-rotating-bucket-of-liquid.38112/
    """
    return  density * angular_speed(speed)**2 * radius**2

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for acceleration
"""
from typing import Annotated

from UliEngineering.EngineerIO.Length import normalize_length
from UliEngineering.EngineerIO.Decorators import normalize_args, returns_unit
from UliEngineering.EngineerIO.Types import NormalizedComputable
from UliEngineering.Physics.Density import normalize_density_kg_per_m3
from UliEngineering.Physics.Frequency import normalize_frequency, normalize_rpm
from UliEngineering.Physics.Mass import normalize_mass_grams
from UliEngineering.Units import Hz
import numpy as np

__all__ = ["rpm_to_Hz", "rpm_to_rps", "hz_to_rpm", "angular_speed",
           "rotation_linear_speed", "centrifugal_force"]


RotationFrequency = Annotated[NormalizedComputable, normalize_frequency]
RotationRateRpm = Annotated[NormalizedComputable, normalize_rpm]
LengthMeters = Annotated[NormalizedComputable, normalize_length]
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

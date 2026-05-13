#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for acceleration."""
from UliEngineering.EngineerIO.Length import LengthMeters
from UliEngineering.EngineerIO.Decorators import normalize_args, returns_unit
from UliEngineering.EngineerIO.Types import NormalizedComputable
from UliEngineering.Physics.Density import DensityKgPerM3
from UliEngineering.Physics.Frequency import normalize_frequency, normalize_rpm, FrequencyHz, RotationFrequency, RotationRate
from UliEngineering.Physics.Mass import MassGrams
import numpy as np

__all__ = ["rpm_to_Hz", "rpm_to_rps", "hz_to_rpm", "angular_speed",
           "rotation_linear_speed", "centrifugal_force"]

@returns_unit("Hz")
@normalize_args
def rpm_to_Hz(rpm_value: RotationRate) -> NormalizedComputable:
    """
    Compute the rotational speed in Hz given the rotational speed in rpm.

    Parameters.
    ----------
    rpm_value : RotationRate
        Rotational speed in rpm.

    Returns
    -------
    NormalizedComputable
        Rotational speed in Hz.
    
    """
    return rpm_value

@returns_unit("rpm")
@normalize_args
def hz_to_rpm(speed: FrequencyHz) -> NormalizedComputable:
    """
    Compute the rotational speed in rpm given the rotational speed in Hz.

    Parameters.
    ----------
    speed : FrequencyHz
        Rotational speed in Hz.

    Returns
    -------
    NormalizedComputable
        Rotational speed in rpm.
    
    """
    return speed * 60.

rpm_to_rps = rpm_to_Hz

@returns_unit("1/s")
@normalize_args
def angular_speed(speed: RotationFrequency) -> NormalizedComputable:
    """
    Compute Ω, the angular speed of a centrifugal system.

    Parameters.
    ----------
    speed : RotationFrequency
        Rotational frequency in Hz.

    Returns
    -------
    NormalizedComputable
        Angular speed in rad/s.
    
    """
    return 2*np.pi*speed

@returns_unit("m/s")
@normalize_args
def rotation_linear_speed(radius: LengthMeters, speed: RotationFrequency) -> NormalizedComputable:
    """
    Compute the linear speed at a given radius for a centrifugal system rotating at speed.

    Parameters.
    ----------
    radius : LengthMeters
        Radius from the center of rotation in meters.
    speed : RotationFrequency
        Rotational frequency in Hz.

    Returns
    -------
    NormalizedComputable
        Linear speed in m/s.
    
    """
    return radius * angular_speed(speed)

@returns_unit("N")
@normalize_args
def centrifugal_force(radius: LengthMeters, speed: RotationFrequency, mass: MassGrams) -> NormalizedComputable:
    """
    Compute the centrifugal force of a mass rotating at speed at radius.

    Parameters.
    ----------
    radius : LengthMeters
        Radius from the center of rotation in meters.
    speed : RotationFrequency
        Rotational frequency in Hz.
    mass : MassGrams
        Mass in grams.

    Returns
    -------
    NormalizedComputable
        Centrifugal force in Newtons.
    
    """
    mass = mass / 1000.0 # mass needs to be Kilograms TODO Improve
    return mass * angular_speed(speed)**2 * radius

@returns_unit("Pa")
@normalize_args
def rotating_liquid_pressure(density: DensityKgPerM3, speed: RotationFrequency, radius: LengthMeters) -> NormalizedComputable:
    """
    Compute the pressure in a body of liquid (relative to the steady-state pressure).
    
    The calculation does not include gravity.

    Also see https://www.youtube.com/watch?v=kIH7wEq3H-M.
    Also see https://www.physicsforums.com/threads/pressure-of-a-rotating-bucket-of-liquid.38112/.

    Parameters
    ----------
    density : DensityKgPerM3
        Density of the liquid in kg/m³.
    speed : RotationFrequency
        Rotational frequency in Hz.
    radius : LengthMeters
        Radius from the center of rotation in meters.

    Returns
    -------
    NormalizedComputable
        Pressure in Pascals.
    
    """
    return  density * angular_speed(speed)**2 * radius**2

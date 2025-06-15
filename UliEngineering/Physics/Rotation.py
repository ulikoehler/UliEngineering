#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for acceleration
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from UliEngineering.Units import Unit
import numpy as np

__all__ = ["rpm_to_Hz", "rpm_to_rps", "hz_to_rpm", "angular_speed",
           "rotation_linear_speed", "centrifugal_force"]

@returns_unit("Hz")
@normalize_numeric_args
def rpm_to_Hz(rpm: Unit("rpm")):
    """
    Compute the rotational speed in Hz given the rotational speed in rpm
    """
    return rpm / 60.

@returns_unit("rpm")
@normalize_numeric_args
def hz_to_rpm(speed: Unit("Hz")):
    """
    Compute the rotational speed in rpm given the rotational speed in Hz
    """
    return speed * 60.

rpm_to_rps = rpm_to_Hz

@returns_unit("1/s")
@normalize_numeric_args
def angular_speed(speed: Unit("Hz")):
    """
    Compute Ω, the angular speed of a centrifugal system
    """
    return 2*np.pi*speed

@returns_unit("m/s")
@normalize_numeric_args
def rotation_linear_speed(radius: Unit("m"), speed: Unit("Hz")):
    """
    Compute the linear speed at a given [radius] for a centrifugal system rotating at [speed].
    """
    return radius * angular_speed(speed)

@returns_unit("N")
@normalize_numeric_args
def centrifugal_force(radius: Unit("m"), speed: Unit("Hz"), mass: Unit("g")):
    """
    Compute the centrifugal force of a [mass] rotation at [speed] at radius [radius]
    """
    mass = mass / 1000.0 # mass needs to be Kilograms TODO Improve
    return mass * angular_speed(speed)**2 * radius

@returns_unit("Pa")
@normalize_numeric_args
def rotating_liquid_pressure(density: Unit("kg/m³"), speed: Unit("Hz"), radius: Unit("m")):
    """
    Compute the pressure in a body of liquid (relative to the steady-state pressure)
    The calculation does not include gravity.

    Also see https://www.youtube.com/watch?v=kIH7wEq3H-M
    Also see https://www.physicsforums.com/threads/pressure-of-a-rotating-bucket-of-liquid.38112/
    """
    return  density * angular_speed(speed)**2 * radius**2

#!/usr/bin/env python3
"""
Utilities to compute the power of a device
"""

__all__ = ["current_by_power", "power_by_current_and_voltage"]

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit


@normalize_numeric_args
@returns_unit("A")
def current_by_power(power="25 W", voltage="230 V"):
    """
    Given a device's power (or RMS power) and the voltage (or RMS voltage)
    it runs on, compute how much current it will draw.
    """
    return power / voltage

@normalize_numeric_args
@returns_unit("W")
def power_by_current_and_voltage(current="1.0 A", voltage="230 V"):
    """
    Given a device's current (or RMS current) and the voltage (or RMS current)
    it runs on, compute its power
    """
    return current * voltage

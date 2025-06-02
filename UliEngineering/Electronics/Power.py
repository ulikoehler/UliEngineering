#!/usr/bin/env python3
"""
Utilities to compute the power of a device
"""
from UliEngineering.EngineerIO import normalize_numeric, normalize_numeric_args
from UliEngineering.Units import Unit

__all__ = ["current_by_power", "power_by_current_and_voltage"]

@normalize_numeric_args
def current_by_power(power="25 W", voltage="230 V") -> Unit("A"):
    """
    Given a device's power (or RMS power) and the voltage (or RMS voltage)
    it runs on, compute how much current it will draw.
    """
    return power / voltage


@normalize_numeric_args
def power_by_current_and_voltage(current="1.0 A", voltage="230 V") -> Unit("W"):
    """
    Given a device's current (or RMS current) and the voltage (or RMS current)
    it runs on, compute its power
    """
    return current * voltage

#!/usr/bin/env python3
"""Utilities to compute the power of a device."""

from UliEngineering.EngineerIO.Decorators import returns_unit
from .Diode import normalize_power, PowerW, normalize_current, CurrentA, normalize_voltage, VoltageV

__all__ = ["current_by_power", "power_by_current_and_voltage"]


@returns_unit("A")
def current_by_power(power: PowerW = "25 W", voltage: VoltageV = "230 V"):
    """Compute how much current a device will draw given its power and voltage.

    Compute how much current a device will draw given its power (or RMS
    power) and the voltage (or RMS voltage) it runs on.

    Parameters
    ----------
    power : PowerW
        Power in Watts. Default is "25 W".
    voltage : VoltageV
        Voltage in Volts. Default is "230 V".

    Returns
    -------
    float
        Current in Amperes.
    """
    power = normalize_power(power) if isinstance(power, str) else power
    voltage = normalize_voltage(voltage) if isinstance(voltage, str) else voltage
    return power / voltage

@returns_unit("W")
def power_by_current_and_voltage(current: CurrentA = "1.0 A", voltage: VoltageV = "230 V"):
    """
    Compute the power of a device given its current and voltage.

    Compute the power of a device given its current (or RMS current) and the
    voltage (or RMS current) it runs on.

    Parameters
    ----------
    current : CurrentA
        Current in Amperes. Default is "1.0 A".
    voltage : VoltageV
        Voltage in Volts. Default is "230 V".

    Returns
    -------
    float
        Power in Watts.
    """
    current = normalize_current(current) if isinstance(current, str) else current
    voltage = normalize_voltage(voltage) if isinstance(voltage, str) else voltage
    return current * voltage

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for computing different aspects and complexities of voltage dividers."""
__all__ = ["zener_diode_power_dissipation"]

from UliEngineering.EngineerIO.Decorators import returns_unit
from .Diode import normalize_voltage, VoltageV, normalize_current, CurrentA


@returns_unit("W")
def zener_diode_power_dissipation(zener_voltage: VoltageV, current: CurrentA):
    """
    Compute the power dissipated in a zener diode given the zener voltage and
    the current through it.

    This is based on an ideal zener diode model and does not take into account
    the zener resistance or the zener knee voltage.
    """
    zener_voltage = normalize_voltage(zener_voltage) if isinstance(zener_voltage, str) else zener_voltage
    current = normalize_current(current) if isinstance(current, str) else current
    return zener_voltage * current
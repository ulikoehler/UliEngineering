#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing different aspects and complexities of voltage dividers
"""
from UliEngineering.EngineerIO import normalize_numeric_args, returns_unit

__all__ = ["zener_diode_power_dissipation"]

@returns_unit("W")
@normalize_numeric_args
def zener_diode_power_dissipation(zener_voltage, current):
    """
    Compute the power dissipated in a zener diode
    given the zener voltage and the current through it.

    This is based on an ideal zener diode model and does not take into account
    the zener resistance or the zener knee voltage.
    """
    return zener_voltage * current
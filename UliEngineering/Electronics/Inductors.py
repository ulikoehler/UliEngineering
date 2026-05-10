#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility to calculate inductors."""

from typing import Annotated

import numpy as np
from UliEngineering.EngineerIO.Decorators import returns_unit
from .Diode import normalize_voltage, VoltageV
from .Filter import normalize_inductance, InductanceH

__all__ = ["ideal_inductor_current_change_rate"]

@returns_unit("A/s")
def ideal_inductor_current_change_rate(inductance: InductanceH, voltage: VoltageV):
    """
    Compute the rise or fall rate of current in an ideal inductor,
    if there's [voltage] across it.

    Parameters
    ----------
    inductance: number or Engineer string
        The inductance in Henrys
    voltage: number or Engineer string
        The voltage across the inductor

    """
    inductance = normalize_inductance(inductance) if isinstance(inductance, str) else inductance
    voltage = normalize_voltage(voltage) if isinstance(voltage, str) else voltage
    return np.divide(voltage, inductance)  # Returns inf when inductance is zero
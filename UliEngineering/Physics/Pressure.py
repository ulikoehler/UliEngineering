#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = ["pascal_to_bar", "bar_to_pascal", "barlow_tangential"]

def pascal_to_bar(pressure: Unit("Pa")) -> Unit("bar"):
    """
    Convert the pressure in pascal to the pressure in bar
    """
    pressure = normalize_numeric(pressure)
    return pressure*1e-5

def bar_to_pascal(pressure: Unit("bar")) -> Unit("Pa"):
    """
    Convert the pressure in bar to the pressure in Pascal
    """
    pressure = normalize_numeric(pressure)
    return pressure*1e5

def barlow_tangential(outer_diameter: Unit("m"), inner_diameter: Unit("m"), pressure: Unit("Pa")) -> Unit("Pa"):
    """
    Compute the tangential stress of a pressure vessel at [pressure] using Barlow's formula for thin-walled tubes.

    Note that this formula only applies for (outer_diameter/inner_diameter) < 1.2 !
    (this assumption is not checked). Otherwise, the stress distribution will be too uneven.
    """
    outer_diameter = normalize_numeric(outer_diameter)
    inner_diameter = normalize_numeric(inner_diameter)
    pressure = normalize_numeric(pressure)
    dm = (outer_diameter + inner_diameter) / 2
    s = (outer_diameter - inner_diameter) / 2
    return pressure * dm / (2 * s)

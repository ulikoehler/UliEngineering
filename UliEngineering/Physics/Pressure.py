#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np

__all__ = ["pascal_to_bar", "bar_to_pascal"]

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

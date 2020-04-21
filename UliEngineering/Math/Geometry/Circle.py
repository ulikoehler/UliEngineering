#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle geometry functions
"""
import math
from UliEngineering.EngineerIO import normalize_numeric, Unit

__all__ = [
    "circle_area", "circle_circumference"
]

def circle_area(radius) -> Unit("m²"):
    """
    Compute the enclosed area of a circle from its radius
    """
    radius = normalize_numeric(radius)
    return math.pi * radius**2

def circle_circumference(radius) -> Unit("m"):
    """
    Compute the circumference of a circle from its radius
    """
    radius = normalize_numeric(radius)
    return 2. * math.pi * radius

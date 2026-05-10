#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circle geometry functions."""
import math

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Types import NormalizableArgument

__all__ = [
    "circle_area", "circle_circumference"
]

@returns_unit("m²")
def circle_area(radius: NormalizableArgument):
    """Compute the enclosed area of a circle from its radius."""
    radius = normalize_numeric(radius) if isinstance(radius, str) else radius
    return math.pi * radius**2

@returns_unit("m")
def circle_circumference(radius: NormalizableArgument):
    """Compute the circumference of a circle from its radius."""
    radius = normalize_numeric(radius) if isinstance(radius, str) else radius
    return 2. * math.pi * radius

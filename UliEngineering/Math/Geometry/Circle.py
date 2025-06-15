#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle geometry functions
"""
import math

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "circle_area", "circle_circumference"
]

@normalize_numeric_args
@returns_unit("m²")
def circle_area(radius):
    """
    Compute the enclosed area of a circle from its radius
    """
    return math.pi * radius**2

@normalize_numeric_args
@returns_unit("m")
def circle_circumference(radius):
    """
    Compute the circumference of a circle from its radius
    """
    return 2. * math.pi * radius

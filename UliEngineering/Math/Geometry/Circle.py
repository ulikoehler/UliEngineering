#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle geometry functions
"""
import math
from UliEngineering.EngineerIO import normalize_numeric

__all__ = [
    "circle_area", "circle_area_from_diameter", "circle_circumference", "circle_circumference", "circle_circumference_from_diameter"
]

def circle_area(radius):
    """
    Compute the enclosed area of a circle from its radius
    """
    radius = normalize_numeric(radius)
    return math.pi * radius**2

def circle_area_from_diameter(diameter):
    """
    Compute the enclosed area of a circle from its diameter
    """
    diameter = normalize_numeric(diameter)
    return math.pi * diameter**2 / 4.0

def circle_circumference(radius):
    """
    Compute the circumference of a circle from its radius
    """
    radius = normalize_numeric(radius)
    return 2. * math.pi * radius

def circle_circumference_from_diameter(diameter):
    """
    Compute the circumference of a circle from its diameter
    """
    diameter = normalize_numeric(diameter)
    return math.pi * diameter
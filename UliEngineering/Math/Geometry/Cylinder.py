#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry functions for cylinders and hollow cylinders
"""
import math
from UliEngineering.EngineerIO import normalize_numeric, Unit
from .Circle import circle_area

__all__ = [
    "cylinder_volume", "cylinder_side_surface_area", "cylinder_surface_area"
]

def cylinder_volume(radius, height) -> Unit("m³"):
    """
    Compute the volume of a cylinder by its radius and height
    """
    radius = normalize_numeric(radius)
    height = normalize_numeric(height)
    return math.pi * (radius**2) * height

def cylinder_side_surface_area(radius, height) -> Unit("m²"):
    """
    Compute the surface area of the side (also called ")
    """
    radius = normalize_numeric(radius)
    height = normalize_numeric(height)
    return 2 * math.pi * radius * height

def cylinder_surface_area(radius, height) -> Unit("m²"):
    """
    Compute the surface area (side + top + bottom)
    """
    radius = normalize_numeric(radius)
    height = normalize_numeric(height)
    return cylinder_side_surface_area(radius, height) + 2 * circle_area(radius)
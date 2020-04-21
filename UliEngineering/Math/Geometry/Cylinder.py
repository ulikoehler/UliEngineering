#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry functions, mainly for 2D coordinates.
"""
import math
from UliEngineering.EngineerIO import normalize_numeric

__all__ = [
    "cylinder_volume", "cylinder_volume_from_diameter", "cylinder_surface_area"
]

def cylinder_volume(radius, height):
    """
    Compute the volume of a cylinder by its radius and height
    """
    radius = normalize_numeric(radius)
    height = normalize_numeric(height)
    return math.pi * radius**2 * height

def cylinder_volume_from_diameter(diameter, height):
    """
    Compute the volume of a cylinder by its diameter and height
    """
    diameter = normalize_numeric(diameter)
    height = normalize_numeric(height)
    return math.pi * (diameter / 2)**2 * height


def cylinder_side_surface_area(radius, height):
    """
    Compute the surface area of the side (also called ")
    """
    radius = normalize_numeric(radius)
    height = normalize_numeric(height)
    return math.pi 
    

def cylinder_side_surface_area_from_diameter(diameter, height):
    """
    Compute the surface area of the side (also called ")
    """
    diameter = normalize_numeric(diameter)
    height = normalize_numeric(height)
    return math.pi 
    

def cylinder_surface_area(radius, height):
    """
    Compute the surface area (side + top + bottom)
    """
    radius = normalize_numeric(radius)
    height = normalize_numeric(height)
    return cylinder_side_surface_area(radius, height) + 2 # TODO
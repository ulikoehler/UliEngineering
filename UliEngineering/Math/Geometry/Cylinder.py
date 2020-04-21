#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry functions for cylinders and hollow cylinders
"""
import math
from UliEngineering.EngineerIO import normalize_numeric, Unit
from .Circle import circle_area
import numpy as np

__all__ = [
    "cylinder_volume", "cylinder_side_surface_area", "cylinder_surface_area",
    "hollow_cylinder_volume", "hollow_cylinder_inner_radius_by_volume"
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

def hollow_cylinder_volume(outer_radius, inner_radius, height) -> Unit("m³"):
    """
    Compute the volume of a hollow cylinder by its height and the inner and outer radii
    """
    return cylinder_volume(outer_radius, height) - cylinder_volume(inner_radius, height)


def hollow_cylinder_inner_radius_by_volume(outer_radius, volume, height) -> Unit("m"):
    """
    Given the outer radius, the height and the inner radius of a hollow cylinder,
    compute the inner radius
    """
    outer_radius = normalize_numeric(outer_radius)
    volume = normalize_numeric(volume)
    height = normalize_numeric(height)
    # Wolfram Alpha: solve V=(pi*o²*h)-(pi*i²*h) for i
    term1 = np.pi*height*(outer_radius**2)-volume
    # Due to rounding errors etc, term1 might become negative.
    # This will lead to sqrt(-x) => NaN but we actually treat it as a zero result
    if term1 < 0.:
        return 0
    # Default case
    return np.sqrt(term1)/(np.sqrt(np.pi) * np.sqrt(height))

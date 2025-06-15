#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry functions for cylinders and hollow cylinders
"""
import math
from .Circle import circle_area
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np

__all__ = [
    "cylinder_volume", "cylinder_side_surface_area", "cylinder_surface_area",
    "hollow_cylinder_volume", "hollow_cylinder_inner_radius_by_volume",
    "cylinder_weight_by_diameter", "cylinder_weight_by_radius",
    "cylinder_weight_by_cross_sectional_area"
]

@normalize_numeric_args
@returns_unit("m³")
def cylinder_volume(radius, height):
    """
    Compute the volume of a cylinder by its radius and height
    """
    return math.pi * (radius**2) * height

@normalize_numeric_args
@returns_unit("m²")
def cylinder_side_surface_area(radius, height):
    """
    Compute the surface area of the side (also called ")
    """
    return 2 * math.pi * radius * height

@normalize_numeric_args
@returns_unit("m²")
def cylinder_surface_area(radius, height):
    """
    Compute the surface area (side + top + bottom)
    """
    return cylinder_side_surface_area(radius, height) + 2 * circle_area(radius)

@normalize_numeric_args
@returns_unit("m³")
def hollow_cylinder_volume(outer_radius, inner_radius, height):
    """
    Compute the volume of a hollow cylinder by its height and the inner and outer radii
    """
    return cylinder_volume(outer_radius, height) - cylinder_volume(inner_radius, height)

@normalize_numeric_args
def cylinder_weight_by_diameter(diameter, length, density=8000):
    """
    Compute the weight of a cylinder by its diameter, length and density.
    The density is in kg/m³, the diameter and length must be given in mm.

    The default density is an approximation for steel.
    """
    return cylinder_volume(diameter/2., length) * density

@normalize_numeric_args
def cylinder_weight_by_radius(radius, length, density=8000):
    """
    Compute the weight of a cylinder by its radius, length and density.
    The density is in kg/m³, the radius and length must be given in mm.

    The default density is an approximation for steel.
    """
    return cylinder_volume(radius, length) * density

@normalize_numeric_args
def cylinder_weight_by_cross_sectional_area(area, length, density=8000):
    """
    Compute the weight of a cylinder by its cross-sectional area, length and density.
    The density is in kg/m³, the area and length must be given in mm² and mm.

    The default density is an approximation for steel.
    """
    return area * length * density

@normalize_numeric_args
@returns_unit("m")
def hollow_cylinder_inner_radius_by_volume(outer_radius, volume, height):
    """
    Given the outer radius, the height and the inner radius of a hollow cylinder,
    compute the inner radius
    """
    # Wolfram Alpha: solve V=(pi*o²*h)-(pi*i²*h) for i
    term1 = np.pi*height*(outer_radius**2)-volume
    # Due to rounding errors etc, term1 might become negative.
    # This will lead to sqrt(-x) => NaN but we actually treat it as a zero result
    if term1 < 0.:
        return 0
    # Default case
    return np.sqrt(term1)/(np.sqrt(np.pi) * np.sqrt(height))

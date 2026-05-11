#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry functions for cylinders and hollow cylinders."""
import math
from .Circle import circle_area
from UliEngineering.EngineerIO.Decorators import normalize_args, returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Types import NormalizableArgument
from UliEngineering.Physics.Density import DensityKgPerM3
import numpy as np

__all__ = [
    "cylinder_volume", "cylinder_side_surface_area", "cylinder_surface_area",
    "hollow_cylinder_volume", "hollow_cylinder_inner_radius_by_volume",
    "cylinder_weight_by_diameter", "cylinder_weight_by_radius",
    "cylinder_weight_by_cross_sectional_area"
]

@returns_unit("m³")
def cylinder_volume(radius: NormalizableArgument, height: NormalizableArgument):
    """Compute the volume of a cylinder by its radius and height."""
    radius = normalize_numeric(radius) if isinstance(radius, str) else radius
    height = normalize_numeric(height) if isinstance(height, str) else height
    return math.pi * (radius**2) * height

@returns_unit("m²")
def cylinder_side_surface_area(radius: NormalizableArgument, height: NormalizableArgument):
    """Compute the surface area of the side (also called lateral surface area)."""
    radius = normalize_numeric(radius) if isinstance(radius, str) else radius
    height = normalize_numeric(height) if isinstance(height, str) else height
    return 2 * math.pi * radius * height

@returns_unit("m²")
def cylinder_surface_area(radius: NormalizableArgument, height: NormalizableArgument):
    """Compute the surface area (side + top + bottom)."""
    radius = normalize_numeric(radius) if isinstance(radius, str) else radius
    height = normalize_numeric(height) if isinstance(height, str) else height
    return cylinder_side_surface_area(radius, height) + 2 * circle_area(radius)

@returns_unit("m³")
def hollow_cylinder_volume(outer_radius: NormalizableArgument, inner_radius: NormalizableArgument, height: NormalizableArgument):
    """Compute the volume of a hollow cylinder by its height and the inner and outer radii."""
    outer_radius = normalize_numeric(outer_radius) if isinstance(outer_radius, str) else outer_radius
    inner_radius = normalize_numeric(inner_radius) if isinstance(inner_radius, str) else inner_radius
    height = normalize_numeric(height) if isinstance(height, str) else height
    return cylinder_volume(outer_radius, height) - cylinder_volume(inner_radius, height)

@normalize_args
def cylinder_weight_by_diameter(diameter: NormalizableArgument, length: NormalizableArgument, density: DensityKgPerM3 = 8000):
    """Compute the weight of a cylinder by its diameter, length and density.

    The density is in kg/m³, the diameter and length must be given in mm.
    The default density is an approximation for steel.
    """
    return cylinder_volume(diameter/2., length) * density

@normalize_args
def cylinder_weight_by_radius(radius: NormalizableArgument, length: NormalizableArgument, density: DensityKgPerM3 = 8000):
    """Compute the weight of a cylinder by its radius, length and density.

    The density is in kg/m³, the radius and length must be given in mm.
    The default density is an approximation for steel.
    """
    return cylinder_volume(radius, length) * density

@normalize_args
def cylinder_weight_by_cross_sectional_area(area: NormalizableArgument, length: NormalizableArgument, density: DensityKgPerM3 = 8000):
    """Compute the weight of a cylinder by its cross-sectional area, length and density.

    The density is in kg/m³, the area and length must be given in mm² and mm.
    The default density is an approximation for steel.
    """
    return area * length * density

@returns_unit("m")
def hollow_cylinder_inner_radius_by_volume(outer_radius: NormalizableArgument, volume: NormalizableArgument, height: NormalizableArgument):
    """Compute the inner radius of a hollow cylinder given outer radius, height, and volume."""
    outer_radius = normalize_numeric(outer_radius) if isinstance(outer_radius, str) else outer_radius
    volume = normalize_numeric(volume) if isinstance(volume, str) else volume
    height = normalize_numeric(height) if isinstance(height, str) else height
    # Wolfram Alpha: solve V=(pi*o²*h)-(pi*i²*h) for i
    term1 = np.pi*height*(outer_radius**2)-volume
    # Due to rounding errors etc, term1 might become negative.
    # This will lead to sqrt(-x) => NaN but we actually treat it as a zero result
    if term1 < 0.:
        return 0
    # Default case
    return np.sqrt(term1)/(np.sqrt(np.pi) * np.sqrt(height))

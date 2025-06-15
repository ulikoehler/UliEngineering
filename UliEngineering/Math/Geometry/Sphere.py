#!/usr/bin/env python3
import math
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = ["sphere_volume_by_radius",
    "sphere_volume_by_diameter",
    "sphere_surface_area_by_radius",
    "sphere_surface_area_by_diameter"]

@normalize_numeric_args
@returns_unit("m³")
def sphere_volume_by_radius(radius):
    """
    Compute the volume of a sphere of a given radius
    """
    return 4./3. * math.pi * radius**3

@normalize_numeric_args
@returns_unit("m³")
def sphere_volume_by_diameter(diameter):
    """
    Compute the volume of a sphere of a given diameter
    """
    return sphere_volume_by_radius(diameter / 2.0)

@normalize_numeric_args
@returns_unit("m²")
def sphere_surface_area_by_radius(radius):
    """
    Compute the surface area of a sphere of a given radius
    """
    return 4. * math.pi * radius**2

@normalize_numeric_args
@returns_unit("m²")
def sphere_surface_area_by_diameter(diameter):
    """
    Compute the surface area of a sphere of a given radius
    """
    return sphere_surface_area_by_radius(diameter / 2.0)
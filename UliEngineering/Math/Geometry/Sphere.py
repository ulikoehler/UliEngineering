#!/usr/bin/env python3
import math
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Types import NormalizableArgument

__all__ = ["sphere_volume_by_radius",
    "sphere_volume_by_diameter",
    "sphere_surface_area_by_radius",
    "sphere_surface_area_by_diameter"]

@returns_unit("m³")
def sphere_volume_by_radius(radius: NormalizableArgument):
    """
    Compute the volume of a sphere of a given radius.

    Parameters
    ----------
    radius : NormalizableArgument
        Radius of the sphere.

    Returns
    -------
    float
        Volume of the sphere in cubic meters.
    """
    radius = normalize_numeric(radius) if isinstance(radius, str) else radius
    return 4./3. * math.pi * radius**3

@returns_unit("m³")
def sphere_volume_by_diameter(diameter: NormalizableArgument):
    """
    Compute the volume of a sphere of a given diameter.

    Parameters
    ----------
    diameter : NormalizableArgument
        Diameter of the sphere.

    Returns
    -------
    float
        Volume of the sphere in cubic meters.
    """
    diameter = normalize_numeric(diameter) if isinstance(diameter, str) else diameter
    return sphere_volume_by_radius(diameter / 2.0)

@returns_unit("m²")
def sphere_surface_area_by_radius(radius: NormalizableArgument):
    """
    Compute the surface area of a sphere of a given radius.

    Parameters
    ----------
    radius : NormalizableArgument
        Radius of the sphere.

    Returns
    -------
    float
        Surface area of the sphere in square meters.
    """
    radius = normalize_numeric(radius) if isinstance(radius, str) else radius
    return 4. * math.pi * radius**2

@returns_unit("m²")
def sphere_surface_area_by_diameter(diameter: NormalizableArgument):
    """
    Compute the surface area of a sphere of a given diameter.

    Parameters
    ----------
    diameter : NormalizableArgument
        Diameter of the sphere.

    Returns
    -------
    float
        Surface area of the sphere in square meters.
    """
    diameter = normalize_numeric(diameter) if isinstance(diameter, str) else diameter
    return sphere_surface_area_by_radius(diameter / 2.0)
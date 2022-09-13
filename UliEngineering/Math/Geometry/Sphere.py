#!/usr/bin/env python3
import math
from UliEngineering.EngineerIO import normalize_numeric, Unit

__all__ = ["sphere_volume_by_radius",
    "sphere_volume_by_diameter"]

def sphere_volume_by_radius(radius) -> Unit("m³"):
    """
    Compute the volume of a sphere of a given radius
    """
    radius = normalize_numeric(radius)
    return 4./3. * math.pi * radius**3

def sphere_volume_by_diameter(diameter) -> Unit("m³"):
    """
    Compute the volume of a sphere of a given diameter
    """
    diameter = normalize_numeric(diameter)
    return sphere_volume_by_radius(diameter / 2.0)

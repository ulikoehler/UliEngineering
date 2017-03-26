#!/usr/bin/env python3
"""
Thread information
"""
from collections import namedtuple

__all__ = ["ThreadParameters", "threads"]

class ThreadParameters(namedtuple("ThreadParameters", ["pitch", "outer_diameter", "core_diameter"])):
    """
    Parameters
    ==========
    pitch:
        Thread pitch in mm
    outer_diameter:
        Outside thread diameter in mm (for exterior thread)
    inner_diameter:
        Inside thread diameter in mm (for exterior thread)
    """
    pass

threads = {
    # DIN 13
    # Source: http://www.gewinde-norm.de/metrisches-iso-gewinde-din-13.htm
    "M1": ThreadParameters(.25, 1., .693),
    "M1.2": ThreadParameters(.25, 1.2, .893),
    "M1.6": ThreadParameters(.35, 1.6, 1.171),
    "M2": ThreadParameters(.40, 2., 1.509),
    "M2.5": ThreadParameters(.45, 2.5, 1.948),
    "M3": ThreadParameters(.5, 3., 2.387),
    "M4": ThreadParameters(.7, 4., 3.141),
    "M5": ThreadParameters(.8, 5., 4.019),
    "M6": ThreadParameters(1., 6., 4.773),
    "M8": ThreadParameters(1.25, 8., 6.466),
    "M10": ThreadParameters(1.5, 10., 8.160),
    "M12": ThreadParameters(1.75, 12., 9.853),
    "M16": ThreadParameters(2., 16., 13.546),
    "M20": ThreadParameters(2.5, 20., 16.933),
    "M24": ThreadParameters(3., 24., 20.319),
    "M36": ThreadParameters(4., 36., 31.093),
    "M42": ThreadParameters(4.5, 42., 36.479),
    "M48": ThreadParameters(5., 48., 41.866),
    "M56": ThreadParameters(5.5, 56., 49.252),
    "M64": ThreadParameters(6., 64., 56.639)
}

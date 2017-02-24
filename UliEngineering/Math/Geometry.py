#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry functions, mainly for 2D coordinates.
"""
import numpy as np

__all__ = ["polygon_lines"]

def polygon_lines(coords, closed=True):
    """
    Given a (n,2) set of XY coordinates as numpy array,
    creates an (n,2,2) array of coordinate pairs.
    If the given coordinate array represents the points of a polygon,
    the return value represents pairs of coordinates to draw lines between
    to obtain the full polygon image.
    
    If closed==True, a line is included between the last and the first point.
    Note that the last->first pair appears first in the list.
    
    Algorithm: http://stackoverflow.com/a/42407359/2597135
    """
    shifted = np.roll(coords, 1, axis=0)
    ret = np.transpose(np.dstack([shifted, coords]), axes=(0, 2, 1))
    return ret if closed else ret[1:]

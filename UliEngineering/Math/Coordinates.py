#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate functions, mainly for 2D coordinates
"""
import numpy as np

__all__ = ["BoundingBox", "polygon_lines"]

class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, points):
        """
        Compute the upright 2D bounding box for a set of
        2D coordinates in a (n,2) numpy array.

        You can access the bbox using the
        (minx, maxx, miny, maxy) members.
        """
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a (n,2), array but it has shape {}".format(
                points.shape))
        if points.shape[0] < 1:
            raise ValueError("Can't compute bounding box for empty coordinates")
        self.minx, self.miny = np.min(points, axis=0)
        self.maxx, self.maxy = np.max(points, axis=0)

    @property
    def width(self):
        """X-axis extent of the bounding box"""
        return self.maxx - self.minx

    @property
    def height(self):
        """Y-axis extent of the bounding box"""
        return self.maxy - self.miny

    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)

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

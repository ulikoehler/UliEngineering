#!/usr/bin/env python3
"""
Slice utilities
"""

__all__ = ["shift_slice"]

def shift_slice(slc, by=0):
    """
    Shift the given slice by <by> index positions.
    Does not take into account the step size of the slice.
    """
    return slice(slc.start + by, slc.stop + by, slc.step)
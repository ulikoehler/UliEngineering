#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to work with wrapped values (e.g. phase angles)
which wrap around at a certain point.
"""
import numpy as np

__all__ = [
    "unwrap"
]

def unwrap(series, wrap_value=2**20, threshold=None):
    """
    Unwrap wrapped values  by compensating for numerical wraps.
    
    Args:
        series: list or np.array
            The input series of wrapped values.
        wrap_value: float
            The value at which wrapping occurs (e.g. 2*pi for angles,
            2**20 for 20-bit counters, etc.)
        threshold: float or None
            The threshold to detect a wrap. If None, it is set to wrap_value / 2.
            A difference larger than this threshold is considered a wrap.
    """
    if threshold is None:
        threshold = wrap_value / 2
        
    # Work with numpy array for performance and handling
    arr = np.array(series, dtype=np.float64)
    
    # Calculate difference between consecutive samples
    # prepend=arr[0] makes the output same length as input, with first diff=0
    diff = np.diff(arr, prepend=arr[0])
    
    # Detect and compensate wraps
    # If diff > threshold, it means value jumped up (e.g. 0 -> max), so we went 'backwards' in continuous space
    # We need to subtract wrap_value from this step
    diff[diff > threshold] -= wrap_value
    
    # If diff < -threshold, it means value jumped down (e.g. max -> 0), so we went 'forwards' in continuous space
    # We need to add wrap_value to this step
    diff[diff < -threshold] += wrap_value
    
    # Reconstruct the signal
    unwrapped = arr[0] + np.cumsum(diff)
    return unwrapped

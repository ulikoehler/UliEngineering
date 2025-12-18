#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to work with wrapped values (e.g. phase angles or encoder counts)
which wrap around at a certain point.
"""
import numpy as np

__all__ = [
    "unwrap",
    "OnlineUnwrapper"
]

class OnlineUnwrapper(object):
    """
    An online unwrapper that can process samples one by one or in chunks.
    Maintains state between calls.
    """
    def __init__(self, wrap_value=2**20, threshold=None):
        """
        Initialize the unwrapper.
        
        Args:
            wrap_value: float
                The value at which wrapping occurs.
            threshold: float or None
                The threshold to detect a wrap. If None, it is set to wrap_value / 2.
        """
        self.wrap_value = wrap_value
        self.threshold = threshold if threshold is not None else wrap_value / 2
        self.last_val = None
        self.correction = 0.0

    def __call__(self, data):
        """
        Unwrap the given data.
        
        Args:
            data: scalar or array-like
                The input value(s) to unwrap.
        
        Returns:
            The unwrapped value(s).
        """
        is_scalar = np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0)
        
        if is_scalar:
            val = float(data)
            if self.last_val is None:
                self.last_val = val
                return val
            
            diff = val - self.last_val
            if diff > self.threshold:
                self.correction -= self.wrap_value
            elif diff < -self.threshold:
                self.correction += self.wrap_value
            
            self.last_val = val
            return val + self.correction
        else:
            arr = np.array(data, dtype=np.float64)
            if len(arr) == 0:
                return arr
            
            if self.last_val is None:
                # First chunk: treat first element as reference (no wrap possible for it)
                prepend_val = arr[0]
                current_correction = 0.0
            else:
                prepend_val = self.last_val
                current_correction = self.correction

            # Calculate diffs including the jump from previous chunk
            diffs = np.diff(arr, prepend=prepend_val)
            
            # Determine wrap corrections for each step
            wrap_adjustments = np.zeros_like(diffs)
            wrap_adjustments[diffs > self.threshold] = -self.wrap_value
            wrap_adjustments[diffs < -self.threshold] = +self.wrap_value
            
            # Cumulative correction for this chunk
            chunk_corrections = np.cumsum(wrap_adjustments)
            
            # Add the carried-over correction from previous chunks
            total_corrections = chunk_corrections + current_correction
            
            out = arr + total_corrections
            
            # Update state
            self.last_val = arr[-1]
            self.correction = total_corrections[-1]
            
            return out

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

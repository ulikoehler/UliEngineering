#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window functions used e.g. for FFTs
"""
import numpy as np

__all__ = ["WindowFunctor", "create_window",
           "create_and_apply_window"]

# Predefined windows

def create_window(size, window_id="blackman", param=None):
    """
    Create a new window numpy array
    param is only used for some windows.

    window_id can also be a function/functor which
    is used to create the window.

    >>> create_window("blackman", 500)
    ... # NumPy array of size 500
    >>> create_window(myfunc, 500, param=3.5)
    ... # result of calling myfunc(500, 3.5)
    """
    if window_id == "blackman":
        return np.blackman(size)
    elif window_id == "bartlett":
        return np.bartlett(size)
    elif window_id == "hamming":
        return np.hamming(size)
    elif window_id == "hanning":
        return np.hanning(size)
    elif window_id == "kaiser":
        return np.kaiser(size, 2.0 if param is None else param)
    elif window_id in ["ones", "none"]:
        return np.ones(size)
    elif callable(window_id):
        return window_id(size, param)
    else:
        raise ValueError(f"Unknown window {window_id}")

def create_and_apply_window(data, window_id="blackman", param=None, inplace=False):
    """
    Create a window suitable for data, multiply it with
    data and return the result

    Parameters
    ----------
    data : numpy array-like
        The data to use. Must be 1D.
    window_id : string or functor
        The name of the window to use.
        See create_window() documentation
    param : number or None
        The parameter used for certain windows.
        See create_window() documentation
    inplace : bool
        If True, data is modified in-place
        If False, data is not modified.
    """
    window = create_window(len(data), window_id, param)
    if inplace:
        data *= window
        return data
    else:
        return data * window

class WindowFunctor(object):
    """
    Initialize a window functor that initializes

    """
    def __init__(self, size, window_id="blackman", param=None):
        """
        Create a new WindowFunctor.
        __init__ initialized the window array

        window_id : string or functor
            The name of the window to use.
            See create_window() documentation
        param : number or None
            The parameter used for certain windows.
            See create_window() documentation
        """
        self.size = size
        self.window = create_window(size, window_id, param=param)

    def __len__(self):
        return self.size

    def __call__(self, data, inplace=False):
        """
        Apply this window to a data array.
        
        Parameters
        ----------
        data : numpy array-like
            The data to apply the window to.
            The length of data must match self.size.
            This is verified.
        inplace : bool
            If True, data is modified in-place
            If False, data is not modified.
        """
        if len(data) != self.size:
            raise ValueError(f"Data size {len(data)} does not match WindowFunctor size {self.size}")
        # Apply
        if inplace:
            data *= self.window
            return data
        else:
            return data * self.window



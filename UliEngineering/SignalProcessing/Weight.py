#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to weight of arrays
"""
import numpy as np

__all__ = ["weigh_halves", "weight_symmetry"]

def weigh_halves(arr, operator=np.sum):
    """
    Split a 1D array into two halves (right in the middle)
    and compute the weight of each half (i.e. sum of all values in that half).
    
    Odd-sized arrays are handled by adding 1/2 of the middle element to each value.
    
    The purpose of this function is to allow to center the "center of weight"
    in sliding window algorithms
    
    Returns (weightLeft, weightReight)
    
    Parameters:
    -----------
    arr : 1D NumPy array
        The array to process.
    operator : unary function
        Alternative operator to summarize the array halves.
        Common choices include np.mean, np.sum or rms from UliEngineering.SignalProcessing.Utils.
    """
    if len(arr) % 2 == 0: # Even array size
        # => We can just split in the middle
        pivot = len(arr) // 2
        return operator(arr[:pivot]), operator(arr[pivot:])
    else:
        # => We can just split in the middle
        pivot = len(arr) // 2
        middle = arr[pivot] / 2
        return operator(arr[:pivot]) + middle, operator(arr[pivot + 1:]) + middle

def weight_symmetry(a, b):
    """
    Given two weights a, b computes a coefficient
    about how equal they are:

    1.0 : Totally equal
    0.0 : Totally different

    The coefficient does not depend on (a + b)
    and is computed using the following formula
    1 - (np.abs(a - b) / (a + b))

    This function is often used like this:
    
    >>> weight_symmetry(*weigh_halves(arr))
    1.0
    """
    return 1 - (np.abs(a - b) / (a + b))

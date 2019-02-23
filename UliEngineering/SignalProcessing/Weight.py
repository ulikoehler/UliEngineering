#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to weight of arrays
"""
import numpy as np

__all__ = ["weigh_halves"]

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

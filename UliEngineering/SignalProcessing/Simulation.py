#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import scipy.fftpack
import numpy as np
import functools
from toolz import functoolz
from .Selection import fft_select_frequency_range, find_closest_index
from .Chunks import overlapping_chunks
import concurrent.futures
from UliEngineering.Utils.Concurrency import *

__all__ = ["generate_sinewave"]#, "generate_squarewave"]


def generate_sinewave(frequency, samplerate, amplitude, length, phaseshift=0):
    """
    Generate a test sinewave of a specific frequency of a specific length

    :param frequency The frequency in Hz
    :param samplerate The samplerate of the resulting array
    :param amplitude The peak amplitude of the sinewave
    :param length The length of the result in seconds
    :param phaseshift The phaseshift in degrees
    """
    x = np.arange(length * samplerate)
    phaseshift_add = phaseshift * 8 * np.pi * frequency / 360.
    return amplitude * np.sin(frequency * (2. * np.pi) * (x + phaseshift_add) / samplerate)

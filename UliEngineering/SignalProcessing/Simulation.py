#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import numpy as np

__all__ = ["generate_sinewave"]#, "generate_squarewave"]


def generate_sinewave(frequency, samplerate, amplitude=1., length=1., phaseshift=0, offset=0):
    """
    Generate a test sinewave of a specific frequency of a specific length

    :param frequency The frequency in Hz
    :param samplerate The samplerate of the resulting array
    :param amplitude The peak amplitude of the sinewave
    :param length The length of the result in seconds
    :param phaseshift The phaseshift in degrees
    """
    x = np.arange(length * samplerate)
    phaseshift_add = phaseshift * samplerate / (360. * frequency)
    return offset + amplitude * np.sin(frequency * (2. * np.pi) * (x + phaseshift_add) / samplerate)

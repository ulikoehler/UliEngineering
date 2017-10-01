#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import numpy as np
import scipy.signal
import functools

__all__ = [
    "generate_sinewave",
    "generate_cosinewave",
    "generate_squarewave",
    "generate_trianglewave",
    "generate_sawtooth",
    "generate_inverse_sawtooth",
]


def _generate_wave(genfn, frequency, samplerate, amplitude=1., length=1., phaseshift=0, offset=0):
    """
    Generate a wave of a specific frequency of a specific length.

    :param frequency A np.sin-like generator function (period shall be 2*pi)
    :param frequency The frequency in Hz
    :param samplerate The samplerate of the resulting array
    :param amplitude The peak amplitude of the sinewave
    :param length The length of the result in seconds
    :param phaseshift The phaseshift in degrees
    """
    x = np.arange(length * samplerate)
    phaseshift_add = phaseshift * samplerate / (360. * frequency)
    return offset + amplitude * genfn(frequency * (2. * np.pi) * (x + phaseshift_add) / samplerate)

generate_sinewave = functools.partial(_generate_wave, np.sin)
generate_cosinewave = functools.partial(_generate_wave, np.cos)
generate_squarewave = functools.partial(_generate_wave, scipy.signal.rect)
generate_trianglewave = functools.partial(_generate_wave,
    functools.partial(scipy.signal.sawtooth, width=0.5))
generate_sawtooth = functools.partial(_generate_wave,
    functools.partial(scipy.signal.sawtooth, width=1))
generate_inverse_sawtooth = functools.partial(_generate_wave,
    functools.partial(scipy.signal.sawtooth, width=0))

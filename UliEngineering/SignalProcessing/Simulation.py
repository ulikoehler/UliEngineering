#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import numpy as np
import functools
from UliEngineering.EngineerIO import normalize_numeric

__all__ = [
    "sine_wave",
    "cosine_wave",
    "square_wave",
    "triangle_wave",
    "sawtooth",
    "inverse_sawtooth",
]


def _generate_wave(genfn, frequency, samplerate, amplitude=1., length=1., phaseshift=0., timedelay=0., offset=0.):
    """
    Generate a wave using a given function of a specific frequency of a specific length.

    :param frequency A np.sin-like generator function (period shall be 2*pi)
    :param frequency The frequency in Hz
    :param samplerate The samplerate of the resulting array
    :param amplitude The peak amplitude of the sinewave
    :param length The length of the result in seconds
    :param timedelay The phaseshift, in seconds (in addition to phaseshift)
    :param phaseshift The phaseshift in degrees (in addition to timedelay)
    """
    # Normalize text values, e.g. "100 kHz" => 100000.0
    frequency = normalize_numeric(frequency)
    samplerate = normalize_numeric(samplerate)
    amplitude = normalize_numeric(amplitude)
    length = normalize_numeric(length)
    phaseshift = normalize_numeric(phaseshift)
    offset = normalize_numeric(offset)
    timedelay = normalize_numeric(timedelay)
    # Perform calculations
    x = np.arange(length * samplerate)
    phaseshift_add = phaseshift * samplerate / (360. * frequency)
    phaseshift_add += timedelay * samplerate
    return offset + amplitude * genfn(frequency * (2. * np.pi) * (x + phaseshift_add) / samplerate)

sine_wave = functools.partial(_generate_wave, np.sin)
cosine_wave = functools.partial(_generate_wave, np.cos)

try:
    import scipy.signal
    square_wave = functools.partial(_generate_wave, scipy.signal.square)
    triangle_wave = functools.partial(_generate_wave,
        functools.partial(scipy.signal.sawtooth, width=0.5))
    sawtooth = functools.partial(_generate_wave,
        functools.partial(scipy.signal.sawtooth, width=1))
    inverse_sawtooth = functools.partial(_generate_wave,
        functools.partial(scipy.signal.sawtooth, width=0))
except ModuleNotFoundError:
    def _error_fn(*args, **kwargs):
        raise NotImplementedError("You need to install scipy to use this function!")
    square_wave = _error_fn
    triangle_wave = _error_fn
    sawtooth = _error_fn
    inverse_sawtooth = _error_fn

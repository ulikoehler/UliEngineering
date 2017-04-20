#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import scipy.fftpack
import numpy as np
import functools
from toolz import functoolz
from .Selection import selectFrequencyRange
from .Chunks import overlapping_chunks
import concurrent.futures
from UliEngineering.Utils.Concurrency import *

__all__ = ["compute_fft", "parallel_fft_reduce", "simple_fft_reduce",
           "fft_cut_dc_artifacts", "fft_cut_dc_artifacts_multi", "generate_sinewave",
           "dominant_frequency", "parallel_fft_reduce_all_results", "fft_frequencies",
           "amplitude_integral", "find_closest_frequency", "serial_fft_reduce",
           "simple_serial_fft_reduce", "simple_parallel_fft_reduce"]

__fft_windows = {
    "blackman": np.blackman,
    "bartlett": np.bartlett,
    "hamming": np.hamming,
    "hanning": np.hanning,
    "kaiser": lambda sz: np.kaiser(sz, 2.0),
    "none": np.ones
}

def fft_frequencies(fftsize, samplerate):
    """Return the frequencies associated to a real-onl FFT array"""
    return np.fft.fftfreq(fftsize)[:fftsize // 2] * samplerate


def compute_fft(y, samplerate, window="blackman"):
    "Compute the real FFT of a dataset and return (x, y) which can directly be visualized using matplotlib etc"
    n = len(y)
    windowArr = __fft_windows[window](n)
    w = scipy.fftpack.fft(y * windowArr)
    w = 2.0 * np.abs(w[:n // 2]) / n  # Perform amplitude normalization
    x = fft_frequencies(n, samplerate)
    return (x, w)


def __fft_reduce_worker(chunkgen, i, window, fftsize, removeDC):
    chunk = chunkgen[i]
    if chunk.size < fftsize:
        raise ValueError("Chunk too small: FFT size {0}, chunk size {1}".format(fftsize, chunk.size))
    yslice = chunk[:fftsize]
    # If enabled, remove DC
    if removeDC:
        yslice = yslice - np.mean(yslice)
    # Compute FFT
    fftresult = scipy.fftpack.fft(yslice * window)
    # Perform amplitude normalization
    return i, np.abs(fftresult[:fftsize // 2])


def sum_reducer(fx, gen):
    "The standard FFT reducer. Sums up all FFT y values."
    return sum(y for _, y in gen)

def parallel_fft_reduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", reducer=sum_reducer, normalize=True, executor=None):
    """
    Perform multiple FFTs on a single dataset, returning the reduction of all FFTs.
    The default reduction method is sum, however any reduction method may be given that
    returns a numeric type that may be normalized (or normalize is set to False).
    Supports optional per-chunk DC offset removal (set removeDC=True).

    Allows flexible y value selection by passing a chunk generator function
    Therefore, y must be a unary function that gets passed arguments from range(numChunks).
    This function must be reentrant and must return a writable version (i.e. if you have
        overlapping chunks or the original array must not be modified for some reason,
        you must return a copy).
    The reducer must not expect to receive the values in any particular order.

    It is recommended to use a shared executor instance. If the executor is set to None,
    a new ThreadPoolExecutor() is used automatically. Using a process-based executor
    is not required as scipy/numpy unlock the GIL during the computationally expensive
    operations.
    """
    if len(chunkgen) == 0:
        raise ValueError("Can't perform FFT on empty chunk generator")
    if executor is None:
        executor = new_thread_executor()
    # Compute common parameters
    window = __fft_windows[window](fftsize)
    fftSum = np.zeros(fftsize // 2)
    # Initialize threadpool
    futures = [
        executor.submit(__fft_reduce_worker, chunkgen, i, window, fftsize, removeDC)
        for i in range(len(chunkgen))
    ]
    # Sum up the results
    x = fft_frequencies(fftsize, samplerate)
    fftSum = reducer(x, (f.result() for f in concurrent.futures.as_completed(futures)))
    # Perform normalization once
    return (x, 2.0 * (fftSum / (len(chunkgen) * fftsize))) if normalize else fftSum


def serial_fft_reduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", reducer=sum_reducer, normalize=True):
    """
    Like parallel_fft_reduce, but performs all operations in serial, i.e. all operations
    are performed on the calling thread
    """
    if len(chunkgen) == 0:
        raise ValueError("Can't perform FFT on empty chunk generator")
    # Compute common parameters
    window = __fft_windows[window](fftsize)
    fftSum = np.zeros(fftsize // 2)
    # Initialize threadpool
    x = fft_frequencies(fftsize, samplerate)
    gen = (
        __fft_reduce_worker(chunkgen, i, window, fftsize, removeDC)
        for i in range(len(chunkgen))
    )
    # Sum up the results
    fftSum = reducer(x, gen)
    # Perform normalization once
    return (x, 2.0 * (fftSum / (len(chunkgen) * fftsize))) if normalize else fftSum



def simple_fft_reduce(fn, arr, samplerate, fftsize, shiftsize=None, nthreads=4, **kwargs):
    """
    Easier interface to (parallel|serial)_fft_reduce that automatically initializes a fixed size chunk generator
    and automatically initializes the executor if no executor is given.

    The shift size is automatically set to fftsize // 4 to account for window function
    masking if no specific value is given.
    """
    shiftsize = fftsize // 4 if shiftsize is None else shiftsize
    chunkgen = overlapping_chunks(arr, fftsize, shiftsize)
    return fn(chunkgen, samplerate, fftsize, **kwargs)

simple_serial_fft_reduce = functools.partial(simple_fft_reduce, serial_fft_reduce)
simple_parallel_fft_reduce = functools.partial(simple_fft_reduce, parallel_fft_reduce)

parallel_fft_reduce_all_results = \
    functools.partial(parallel_fft_reduce, normalize=False, reducer=functoolz.identity)

def fft_cut_dc_artifacts(fx, fy=None, return_idx=False):
    """
    If an FFT contains DC artifacts, i.e. a large value in the first FFT samples,
    this function can be used to remove this area from the FFT value set.
    This function cuts every value up to (but not including the) first local minimum.
    It returns a tuple (x, y)

    Use return_idx=True to return the start index instead of slices
    """
    # Unpack tuple if directly called on the value of
    if fy is None:
        fx, fy = fx
    lastVal = fy[0]
    idx = 0
    # Loop until first local minimum
    for y in fy:
        if y > lastVal:
            if return_idx:
                return idx
            return (fx[idx:], fy[idx:])
        idx += 1
        lastVal = y
    # No minimum found. We can't remove DC offset, so return something non-empty (= consistent)
    if return_idx:
        return 0
    return (fx, fy)

def fft_cut_dc_artifacts_multi(fx, fys, return_idx=False):
    """Remove FFT artifacts for a list of numpy arrays. Resizes all arrays to the same size"""
    idx = max(fft_cut_dc_artifacts(None, fy, return_idx=True) for fy in fys)
    if return_idx:
        return idx
    return fx[idx:], [fy[idx:] for fy in fys]

def dominant_frequency(x, y=None, low=None, high=None):
    """
    Return the frequency with the largest amplitude in a FFT spectrum
    Optionally, a frequency range may be given
    """
    if y is None:  # So we can pass in a FFT result tuple directly
        x, y = x
    # Apply frequency range
    if low is not None or high is not None:
        x, y = selectFrequencyRange(x, y, low=low, high=high)
    return x[np.argmax(y)]


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


def amplitude_integral(fx, fy, low=None, high=None):
    """
    Return the amplitude integral of a frequency-domain signal.
    Optionally, the signal can be filtered directly.
    Call this on a (fx, fy) pair as returned by compute_fft.

    The value at the low boundary is included in the range while the value
    at the high boundary is not.

    :return The amplitude integral value normalized as [amplitude unit] / Hz
    """
    fx, fy = selectFrequencyRange(fx, fy, low=low, high=high)
    # Normalize to [amplitude unit] / Hz
    hz = fx[-1] - fx[0]
    return np.sum(fy) / hz


def find_closest_frequency(fftx, ffty, frequency):
    """
    Find the closest frequency bin and value in a FFT.
    Return (frequency of closest frequency bin, value)
    """
    idx = np.argmin(np.abs(fftx - frequency))
    return fftx[idx], ffty[idx]

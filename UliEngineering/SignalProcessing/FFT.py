#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import warnings
import numpy as np
import functools
from toolz import functoolz
from .Selection import find_closest_index, sorted_range_indices
from .Chunks import overlapping_chunks
from .Window import create_and_apply_window, WindowFunctor
import concurrent.futures
from collections import namedtuple
from UliEngineering.Utils.Concurrency import QueuedThreadExecutor
from UliEngineering.SignalProcessing.Utils import remove_mean


__all__ = ["compute_fft", "parallel_fft_reduce", "simple_fft_reduce", "FFTPoint",
           "fft_cut_dc_artifacts", "fft_cut_dc_artifacts_multi", "fft_frequencies", "FFT",
           "serial_fft_reduce", "simple_serial_fft_reduce", "simple_parallel_fft_reduce"]

# Optional scipy dependency: Use either faster scipy or fallback to numpy
try:
    import scipy.fftpack
    _fft_backend = scipy.fftpack.fft
except ModuleNotFoundError:
    import numpy.fft
    _fft_backend = numpy.fft.fft
    warnings.warn("Using NumPy FFT backend fallback, install scipy for faster FFTs!", RuntimeWarning)

FFTPoint = namedtuple("FFTPoint", ["frequency", "amplitude", "angle"])

class FFT(object):
    """
    FFT result wrapper that allows convenient access to various functions
    """
    def __init__(self, frequencies, amplitudes, angles=None):
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.angles = angles

    def __getitem__(self, arg):
        """
        Select a frequency range:
        fft[1.0:100.0] selects the 1.0 ... 100.0 Hz frequency range
        fft[1.0:] selects everything from 1.0 Hz to the max frequency
        fft[:100.0] selects everything from 1.0 Hz to the max frequency

        fft[10.0] selects (frequency, value, angle) of 
        """
        if isinstance(arg, slice) or isinstance(arg, tuple):
            if isinstance(arg, slice):
                start, end = arg.start, arg.stop
            else: # arg is tuple
                start, end = arg
            startidx, endidx = sorted_range_indices(self.frequencies, start, end)
            # Remove everything except the selected frequency range
            return FFT(
                self.frequencies[startidx:endidx],
                self.amplitudes[startidx:endidx],
                self.angles[startidx:endidx] if self.angles is not None else None
            )
        elif isinstance(arg, (float, int)):
            return self.closest_value(arg)
        else: # Delegate to tuple impl
            raise ValueError("FFT [] operator selects a frequency range([start:stop]) or (value, angle) at the closest frequency ([frequency])! {} is an illegal argument".format(arg))

    def dominant_frequency(self, low=None, high=None):
        """
        Return the frequency with the largest amplitude in a FFT spectrum
        Optionally, a frequency range (low, high) may be given, in which case
        the dominant frequency is only selected from that range.
        """
        # Apply frequency range
        if low is not None or high is not None:
            self = self[low:high]
        return self.frequencies[np.argmax(self.amplitudes)]
    
    def dominant_value(self, low=None, high=None):
        """
        Return the value with the largest amplitude in a FFT spectrum.
        The value is returned as a FFTPoint object.
        Use .frequency, .amplitude and .angle to access
        Optionally, a frequency range (low, high) may be given, in which case
        the dominant frequency is only selected from that range.
        """
        domfreq = self.dominant_frequency(low, high)
        return self.closest_value(domfreq)

    def amplitude_integral(self, low=None, high=None):
        """
        Return the amplitude integral of a frequency-domain signal.
        Optionally, the signal can be filtered directly.
        Call this on a FFT() object as returned by compute_fft.

        The value at the low boundary is included in the range while the value
        at the high boundary is not.

        :return The amplitude integral value normalized as [amplitude unit] / Hz
        """
        filtered = self[low, high]
        # Normalize to [amplitude unit] / Hz
        dHz = filtered.frequencies[-1] - filtered.frequencies[0]
        return np.sum(filtered.amplitudes) / dHz
        
    def closest_frequency(self, frequency):
        """
        Find the closest frequency bin and value in an array of frequencies
        Return (frequency of closest frequency bin, value, angle).
        """
        return self.frequencies[find_closest_index(self.frequencies, frequency)]

    def closest_value(self, frequency):
        """
        Find the closest frequency, value and angle
        Return (frequency of closest frequency bin, value, angle)
        as a FFTPoint object.
        
        Use .frequency, .amplitude and .angle to access
        """
        idx = find_closest_index(self.frequencies, frequency)
        return FFTPoint(
            self.frequencies[idx],
            self.amplitudes[idx],
            self.angles[idx] if self.angles is not None else None
        )

    def cut_dc_artifacts(self, return_idx=False):
        """
        If an FFT contains DC artifacts, i.e. a large value in the first FFT samples,
        this function can be used to remove this area from the FFT value set.
        This function cuts every value up to (but not including the) first local minimum.
        It returns a new FFT object

        Use return_idx=True to return the start index instead of slices
        """
        return fft_cut_dc_artifacts(self, return_idx=return_idx)

def fft_frequencies(fftsize, samplerate):
    """Return the frequencies associated to a real-onl FFT array"""
    return np.fft.fftfreq(fftsize)[:fftsize // 2] * samplerate

def compute_fft(y, samplerate, window="blackman", window_param=None):
    """
    Compute the real FFT of a dataset and return an FFT object which can directly be visualized using matplotlib etc:
    result = compute_fft(...)
    plt.plot(result.frequencies, result.amplitudes)

    The angles are returned as degrees.
    Usually, due to the oscillating phases of spectral leakage,
    it doesn't make sense to visualize the angles directly but to
    select the angle e.g. where the amplitudes have a peak
    """
    n = len(y)
    windowedY = create_and_apply_window(y, window, param=window_param)
    w = _fft_backend(windowedY)[:n // 2]
    w_norm = 2.0 * np.abs(w) / n  # Perform amplitude normalization
    x = fft_frequencies(n, samplerate)
    angles = np.rad2deg(np.angle(w))
    return FFT(x, w_norm, angles)

def __fft_reduce_worker(chunkgen, i, window, fftsize, removeDC):
    chunk = chunkgen[i]
    if chunk.size < fftsize:
        raise ValueError("Chunk too small: FFT size {0}, chunk size {1}".format(fftsize, chunk.size))
    yslice = chunk[:fftsize]
    # If enabled, remove DC
    # Do NOT do this in place as the data might be processed by overlapping FFTs or otherwise
    if removeDC:
        yslice = remove_mean(yslice)
    # Compute FFT
    fftresult = _fft_backend(window(yslice))
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
        executor = QueuedThreadExecutor()
    # Compute common parameters
    window = WindowFunctor(fftsize, window)
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
    if normalize:
        fftSum = fftSum * 2.0 / (len(chunkgen) * fftsize)
    return FFT(x, fftSum, None)


def serial_fft_reduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", reducer=sum_reducer, normalize=True, window_param=None):
    """
    Like parallel_fft_reduce, but performs all operations in serial, i.e. all operations
    are performed on the calling thread
    """
    if len(chunkgen) == 0:
        raise ValueError("Can't perform FFT on empty chunk generator")
    # Compute common parameters
    window = WindowFunctor(fftsize, window, param=window_param)
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
    if normalize:
        fftSum = fftSum * 2.0 / (len(chunkgen) * fftsize)
    return FFT(x, fftSum, None)


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

def fft_cut_dc_artifacts(fft, return_idx=False):
    """
    If an FFT contains DC artifacts, i.e. a large value in the first FFT samples,
    this function can be used to remove this area from the FFT value set.
    This function cuts every value up to (but not including the) first local minimum.
    It returns a tuple (x, y)

    Use return_idx=True to return the start index instead of slices
    """
    # Unpack tuple if directly called on the value of
    lastVal = fft.amplitudes[0]
    idx = 0
    # Loop until first local minimum
    for y in fft.amplitudes:
        if y > lastVal:
            if return_idx:
                return idx
            return FFT(fft.frequencies[idx:], fft.amplitudes[idx:], fft.angles[idx:] if fft.angles is not None else None)
        idx += 1
        lastVal = y
    # No minimum found. We can't remove DC offset, so return something non-empty (= consistent)
    if return_idx:
        return 0
    return fft


def fft_cut_dc_artifacts_multi(fx, fys, return_idx=False):
    """Remove FFT artifacts for a list of numpy arrays. Resizes all arrays to the same size"""
    idx = max(fft_cut_dc_artifacts(FFT(None, fy), return_idx=True) for fy in fys)
    if return_idx:
        return idx
    return fx[idx:], [fy[idx:] for fy in fys]


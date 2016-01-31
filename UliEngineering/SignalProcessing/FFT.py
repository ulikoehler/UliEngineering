#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import scipy.fftpack
import numpy as np
import numpy.fft
import functools
import os
from toolz import functoolz
from .Selection import selectFrequencyRange
from .Chunks import overlapping_chunks
import concurrent.futures

__all__ = ["computeFFT", "parallelFFTReduce", "simpleParallelFFTReduce",
           "cutFFTDCArtifacts", "cutFFTDCArtifactsMulti",
           "dominantFrequency", "parallelFFTReduceAllResults"]

__fft_windows = {
    "blackman": np.blackman,
    "bartlett": np.bartlett,
    "hamming": np.hamming,
    "hanning": np.hanning,
    "kaiser": lambda sz: np.kaiser(sz, 2.0),
    "none": np.ones
}


def computeFFT(y, samplerate, window="blackman"):
    "Compute the real FFT of a dataset and return (x, y) which can directly be visualized using matplotlib etc"
    n = len(y)
    windowArr = __fft_windows[window](n)
    w = scipy.fftpack.fft(y * windowArr)
    w = 2.0 * np.abs(w[:n / 2]) / samplerate # Perform amplitude normalization
    x = np.linspace(0.0, samplerate / 2, n / 2)
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
    return np.abs(fftresult[:fftsize / 2])

def parallelFFTReduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", reducer=sum, normalize=True, executor=None):
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
    The 

    It is recommended to use a shared executor instance. If the executor is set to None,
    a new ThreadPoolExecutor() is used automatically. Using a process-based executor
    is not required as scipy/numpy unlock the GIL during the computationally expensive
    operations.
    """
    if len(chunkgen) == 0:
        raise ValueError("Can't perform FFT on empty chunk generator")
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1)
    # Compute common parameters
    window = __fft_windows[window](fftsize)
    fftSum = np.zeros(fftsize / 2)
    # Initialize threadpool
    futures = [
        executor.submit(__fft_reduce_worker, chunkgen, i, window, fftsize, removeDC)
        for i in range(len(chunkgen))
    ]
    # Sum up the results
    x = np.linspace(0.0, samplerate / 2, fftsize / 2)
    fftSum = reducer((f.result() for f in concurrent.futures.as_completed(futures)))
    # Perform normalization once
    return (x, 2.0 * (fftSum / (len(chunkgen) * samplerate))) if normalize else fftSum


def simpleParallelFFTReduce(arr, samplerate, fftsize, shiftsize=None, nthreads=4, chunkfunc=None, **kwargs):
    """
    Easier interface to parallelFFTSum that automatically initializes a fixed size chunk generator
    and automatically initializes the executor if no executor is given.

    The shift size is automatically set to fftsize // 4 to account for window function
    masking if no specific value is given.
    """
    shiftsize = fftsize // 4 if shiftsize is None else shiftsize
    chunkgen = overlapping_chunks(arr, fftsize, shiftsize)
    return parallelFFTReduce(chunkgen, samplerate, fftsize, **kwargs)

parallelFFTReduceAllResults = \
    functools.partial(parallelFFTReduce, normalize=False, reducer=functoolz.identity)

def cutFFTDCArtifacts(fx, fy=None, return_idx=False):
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

def cutFFTDCArtifactsMulti(fx, fys, return_idx=False):
    """Remove FFT artifacts for a list of numpy arrays. Resizes all arrays to the same size"""
    idx = max(cutFFTDCArtifacts(None, fy, return_idx=True) for fy in fys)
    if return_idx:
        return idx
    return fx[idx:], [fy[idx:] for fy in fys]

def dominantFrequency(x, y=None, low=None, high=None):
    """
    Return the frequency with the largest amplitude in a FFT spectrum
    Optionally, a frequency range may be given
    """
    if y is None:  # So we can pass in a FFT result tuple directly
        x, y = x
    # Apply frequency range
    if low is not None or high is not None:
        xv, yv = selectFrequencyRange(x, y, low=low, high=high)
    else:
        xv, yv = x, y
    return xv[np.argmax(yv)]

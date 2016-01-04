#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import scipy.fftpack
import numpy as np
import numpy.fft
import functools
import concurrent.futures

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
    x = np.linspace(0.0, samplerate/2, n/2)
    return (x, w)

def __chunkedFFTWorker(y, c, fftsize, windowArr, removeDC):
    yslice = y(c)
    # If enabled, remove DC
    if removeDC:
        yslice -= np.mean(yslice)
    # Compute FFT
    w = scipy.fftpack.fft(yslice * windowArr)
    # Perform amplitude normalization
    return np.abs(w[:fftsize / 2])

def parallelFFTSum(executor, y, numChunks, samplerate, fftsize, removeDC=False, window="blackman"):
    """
    Perform multiple FFTs on a single dataset, returning the sum of all FFTs.
    Supports optional per-chunk DC offset removal (set removeDC=True).

    Allows flexible y value selection by passing a function that gets the nth FFT chunk.
    Therefore, y must be a unary function that gets passed arguments from range(numChunks).
    This function must be reentrant and must return a writable version (i.e. if you have
        overlapping chunks or the original array must not be modified for some reason,
        you must return a copy).
    """
    # Compute common parameters
    windowArr = __fft_windows[window](fftsize)
    fftSum = np.zeros(fftsize / 2)
    # Initialize threadpool
    futures = [
        executor.submit(__chunkedFFTWorker, y, i,
                        fftsize, windowArr, removeDC)
        for i in range(numChunks)
    ]
    # Sum up the results
    x = np.linspace(0.0, samplerate / 2, fftsize / 2)
    fftSum = sum((f.result() for f in concurrent.futures.as_completed(futures)))
    # Perform normalization once
    return x, 2.0 * (fftSum / numChunks) / samplerate

def cutFFTDCArtifacts(fx, fy=None):
    """
    If an FFT contains DC artifacts, i.e. a large value in the first FFT samples,
    this function can be used to remove this area from the FFT value set.
    This function cuts every value up to (but not including the) first local minimum.
    It returns a tuple (x, y)
    """
    # Unpack tuple if directly called on the value of
    if fy is None:
        fx, fy = fx
    lastVal = fy[0]
    idx = 0
    # Loop until first local minimum
    for y in fy:
        if y > lastVal:
            return (fx[idx:], fy[idx:])
        idx += 1
        lastVal = y
    # No mimum found. We can't remove DC offset, so return something non-empty (= consistent)
    return (fx, fy)

def selectFrequenciesByThreshold(fx, fy, thresh):
    """
    Select frequencies where a specific absolute threshold applies
    Returns an array of frequencies
    """
    return np.asarray([fx[idx] for idx in range(len(fy)) if fy[idx] > thresh])

def dominantFrequency(x, y=None):
    "Return the frequency with the largest amplitude in a FFT spectrum"
    if y is None: # So we can pass in a FFT result tuple directly
        x, y = x
    return x[np.argmax(y)]
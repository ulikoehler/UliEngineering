#!/usr/bin/env python3
"""
Utilities for FFT computation and visualization
"""
import scipy.fftpack
import numpy as np
import numpy.fft
import functools

__fft_windows = {
	"blackman": np.blackman,
	"bartlett": np.bartlett,
	"hamming": np.hamming,
	"hanning": np.hanning,
	"kaiser": lambda sz: np.kaiser(sz, 2.0),
}

def computeFFT(y, samplerate, window="blackman"):
    "Compute the real FFT of a dataset and return (x, y) which can directly be visualized using matplotlib etc"
    windowArr = __fft_windows[window]
    n = len(y)
    w =  scipy.fftpack.fft(y)
    w  = 2.0 * np.abs(w[:n / 2]) / samplerate # Perform amplitude normalization
    x = np.linspace(0.0, samplerate/2, n/2) 
    return (x, w)

def cutFFTDCArtifats(fx, fy):
	"""
	If an FFT contains DC artifacts, i.e. a large value in the first FFT samples,
	this function can be used to remove this area from the FFT value set.
	This function cuts every value up to (but not including the) first local minimum.
	It returns a tuple (x, y)
	"""
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

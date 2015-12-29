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

def __chunkedFFTWork(y, yofs, samplerate, fftsize, windowArr, removeDC):
    yslice = y[yofs:yofs + fftsize].copy()
    # If enabled, remove DC
    if removeDC:
        yslice -= np.mean(yslice)
    # Compute FFT
    w = scipy.fftpack.fft(yslice * windowArr)
    # Perform amplitude normalization
    return 2.0 * np.abs(w[:fftsize / 2]) / samplerate

def chunkedFFTSum(y, samplerate, fftsize, shiftsize=None, removeDC=False, threads=8, window="blackman"):
    """
    Perform multiple FFTs on a single dataset, returning the sum of all FFTs.
    Supports optional per-chunk DC offset removal (set removeDC=True).

    Exploits
    """
    if shiftsize is None:
        shiftsize = int(fftsize / 2)
    # Compute common parameters
    windowArr = __fft_windows[window](fftsize)
    fftSum = np.zeros(fftsize / 2)
    numSlices = 0
    # Initialize threadpool
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        futures = []
        for ofs in range(0, y.shape[0], shiftsize):
            # Skip last (non-full) block
            if ofs + fftsize > y.shape[0]:
                break
            numSlices += 1
            futures.append(
                executor.submit(__chunkedFFTWork, y, ofs, samplerate,
                                fftsize, windowArr, removeDC))
        # Sum up
        fftSum = sum((f.result() for f in concurrent.futures.as_completed(futures)))
    x = np.linspace(0.0, samplerate / 2, fftsize / 2)
    return x, (fftSum / numSlices)


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

def selectFrequenciesByThreshold(fx, fy, thresh):
	"""
	Select frequencies where a specific absolute threshold applies
	Returns an array of frequencies
	"""
	return np.asarray([fx[idx] for idx in range(len(fy)) if fy[idx] > thresh])

def showFrequencyMark(ax, fx, fy, freq):
    """
    Show a vertical mark with a label
    """
    # Render line
    ax.axvline(x)
    # Render text
    textY = min(val + 0.15 * globalMagnitude, .95 * globalMax)
    plt.text(freq, textY,
             '$%.3f\ \\mathrm{Hz}$' % freq,
             bbox=dict(boxstyle='round', facecolor='wheat'),
             verticalalignment='bottom', fontsize = 18, rotation=30)

def dominantFrequency(x, y):
    "Return the frequency with the largest amplitude in a FFT spectrum"
    return x[np.argmax(y)]
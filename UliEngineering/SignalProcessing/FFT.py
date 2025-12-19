#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for FFT computation and visualization
"""
import warnings
import numpy as np
import functools
from .Selection import find_closest_index, sorted_range_indices
from .Chunks import overlapping_chunks
from .Window import create_and_apply_window, WindowFunctor
import concurrent.futures
from collections import namedtuple
from UliEngineering.Utils.Concurrency import QueuedThreadExecutor
from UliEngineering.SignalProcessing.Utils import remove_mean


__all__ = ["compute_fft", "parallel_fft_reduce", "simple_fft_reduce", "FFTPoint",
           "fft_cut_dc_artifacts", "fft_cut_dc_artifacts_multi", "fft_frequencies", "FFT",
           "serial_fft_reduce", "simple_serial_fft_reduce", "simple_parallel_fft_reduce",
           "spectral_power_reducer", "parallel_spectral_power_fft_reduce", "serial_spectral_power_fft_reduce",
           "simple_serial_spectral_power_fft_reduce", "simple_parallel_spectral_power_fft_reduce"]

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

class FFTReductionOverTime(object):
    """Container for per-FFT reduction values over time.

    Attributes
    ----------
    powers : numpy.ndarray
        1D array with one value per FFT chunk containing the reduced power.
    start_indices : numpy.ndarray
        Start sample index of the time slice for each FFT chunk (float).
    end_indices : numpy.ndarray
        End sample index of the time slice for each FFT chunk (float).
    mid_indices : numpy.ndarray
        Middle sample index of the time slice for each FFT chunk (float).
    fftsize : int
        FFT size used to compute the power.
    samplerate : float or None
        Samplerate used (if provided) to compute times; otherwise None.
    start_freq : float or None
        Start frequency used for the band selection.
    end_freq : float or None
        End frequency used for the band selection.
    """
    def __init__(self, powers, start_indices, end_indices, fftsize, samplerate=None, start_freq=None, end_freq=None):
        import numpy as _np
        self.powers = _np.asarray(powers)
        self.start_indices = _np.asarray(start_indices, dtype=float)
        self.end_indices = _np.asarray(end_indices, dtype=float)
        self.mid_indices = (self.start_indices + self.end_indices) / 2.0
        self.fftsize = int(fftsize)
        self.samplerate = None if samplerate is None else float(samplerate)
        self.start_freq = start_freq
        self.end_freq = end_freq
        if not (self.powers.shape[0] == self.start_indices.shape[0] == self.end_indices.shape[0]):
            raise ValueError("powers, start_indices and end_indices must have the same length")

    def __len__(self):
        return self.powers.shape[0]

    def __getitem__(self, idx):
        return (self.start_indices[idx], self.end_indices[idx], self.mid_indices[idx], self.powers[idx])

    def times(self):
        """Return time positions (middle of each FFT) in seconds if samplerate is set, else None."""
        if self.samplerate is None:
            return None
        return self.mid_indices / self.samplerate

    def as_array(self):
        """Return powers as a NumPy array."""
        return self.powers

    def mean(self):
        """Return mean power over time."""
        return float(self.powers.mean())

    def __repr__(self):
        return f"FFTReductionOverTime(len={len(self)}, fftsize={self.fftsize}, start_freq={self.start_freq}, end_freq={self.end_freq})"



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
    # Perform amplitude normalization (use centralized helper)
    w_norm = normalize_fft_reduction(np.abs(w), n, nchunks=1, power=False)
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

def spectral_power_reducer(fx, gen):
    "FFT reducer that computes the sum of squares of the FFT y values."
    return sum(y**2 for _, y in gen)


def normalize_fft_reduction(values, fftsize, nchunks=1, power=False):
    """Normalize FFT reduction results.

    Parameters
    ----------
    values : array_like
        Values to normalize (amplitudes or powers).
    fftsize : int
        FFT size (n)
    nchunks : int
        Number of averaged chunks
    power : bool
        If True, treat `values` as summed powers (squared amplitudes) and
        apply the power normalization. Otherwise, use amplitude normalization.

    Returns
    -------
    numpy.ndarray
        Normalized values
    """
    vals = np.asarray(values)
    if power:
        factor = 4.0 / (nchunks * fftsize * fftsize)
    else:
        factor = 2.0 / (nchunks * fftsize)
    return vals * factor

def parallel_fft_reduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", reducer=sum_reducer, normalize=True, executor=None, window_param=None):
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
    window = WindowFunctor(fftsize, window, param=window_param)
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
        fftSum = normalize_fft_reduction(fftSum, fftsize, len(chunkgen), power=False)
    return FFT(x, fftSum, None)


def serial_fft_reduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", reducer=sum_reducer, normalize=True, window_param=None):
    """
    Serial wrapper that calls the parallel implementation with a single-threaded executor.
    """
    if len(chunkgen) == 0:
        raise ValueError("Can't perform FFT on empty chunk generator")
    executor = QueuedThreadExecutor(nthreads=1)
    return parallel_fft_reduce(chunkgen, samplerate, fftsize, removeDC=removeDC, window=window, reducer=reducer, normalize=normalize, executor=executor, window_param=window_param)


def parallel_spectral_power_fft_reduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", normalize=True, start=0.0, end=None, executor=None, window_param=None):
    """
    Like (parallel|serial)_fft_reduce, but computes a single power value per FFT chunk
    representing the total power inside the requested frequency band.

    Returns a numpy array of length equal to the number of chunks. Each element is the
    spectral power for one FFT (time slot).

    Parameters
    ----------
    start : float or None
        Start frequency (inclusive). Defaults to 0.0.
    end : float or None
        End frequency (exclusive). Defaults to the maximum frequency.
    """
    if len(chunkgen) == 0:
        raise ValueError("Can't perform FFT on empty chunk generator")
    nchunks = len(chunkgen)
    if executor is None:
        executor = QueuedThreadExecutor()
    # Compute frequency array and selection indices
    x = fft_frequencies(fftsize, samplerate)
    startidx, endidx = sorted_range_indices(x, start, end)
    # Convert None to bounds
    startidx = 0 if startidx is None else startidx
    endidx = x.shape[0] if endidx is None else endidx
    # Prepare result array (one value per FFT chunk)
    powers = np.zeros(nchunks)
    # Prepare common window
    windowfun = WindowFunctor(fftsize, window)
    # Submit workers
    futures = [executor.submit(__fft_reduce_worker, chunkgen, i, windowfun, fftsize, removeDC)
               for i in range(nchunks)]
    # As futures complete, compute per-chunk band power and store at correct index
    starts = []
    ends = []
    for f in concurrent.futures.as_completed(futures):
        i, mags = f.result()
        # Compute integral
        p = np.sum(mags[startidx:endidx] ** 2)
        if normalize:
            # For a single FFT, power normalization uses nchunks=1
            p = normalize_fft_reduction(p, fftsize, nchunks=1, power=True)
        powers[i] = p
        # Attempt to obtain the original index slice for time mapping
        try:
            sl = chunkgen.original_indexes(i)
            starts.append((i, float(sl.start)))
            ends.append((i, float(sl.stop)))
        except Exception:
            # Fallback: approximate using chunk index and fftsize
            starts.append((i, float(i)))
            ends.append((i, float(i + fftsize)))
    # Recreate ordered arrays
    starts_arr = np.zeros(len(powers), dtype=float)
    ends_arr = np.zeros(len(powers), dtype=float)
    for i, v in starts:
        starts_arr[i] = v
    for i, v in ends:
        ends_arr[i] = v
    return FFTReductionOverTime(powers, starts_arr, ends_arr, fftsize, samplerate=samplerate, start_freq=start, end_freq=end)


def serial_spectral_power_fft_reduce(chunkgen, samplerate, fftsize, removeDC=False, window="blackman", normalize=True, start=0.0, end=None, window_param=None):
    """
    Like serial_fft_reduce, but computes the average spectral power (amplitude squared) only in
    the requested frequency band. The selection is applied while computing the spectrum.

    Parameters
    ----------
    start : float or None
        Start frequency (inclusive). Defaults to 0.0.
    end : float or None
        End frequency (exclusive). Defaults to the maximum frequency.
    """
    if len(chunkgen) == 0:
        raise ValueError("Can't perform FFT on empty chunk generator")
    # Compute frequency array and selection indices
    x = fft_frequencies(fftsize, samplerate)
    startidx, endidx = sorted_range_indices(x, start, end)
    # Convert None to bounds
    startidx = 0 if startidx is None else startidx
    endidx = x.shape[0] if endidx is None else endidx
    # Prepare accumulation buffer
    n_bins = endidx - startidx
    fftPower = np.zeros(n_bins)
    # Prepare common window
    windowfun = WindowFunctor(fftsize, window, param=window_param)
    # Loop over chunks and compute one power value per chunk
    nchunks = len(chunkgen)
    powers = np.zeros(nchunks)
    starts = []
    ends = []
    for i in range(nchunks):
        _, mags = __fft_reduce_worker(chunkgen, i, windowfun, fftsize, removeDC)
        p = np.sum(mags[startidx:endidx] ** 2)
        if normalize:
            p = normalize_fft_reduction(p, fftsize, nchunks=1, power=True)
        powers[i] = p
        try:
            sl = chunkgen.original_indexes(i)
            starts.append((i, float(sl.start)))
            ends.append((i, float(sl.stop)))
        except Exception:
            starts.append((i, float(i)))
            ends.append((i, float(i + fftsize)))
    starts_arr = np.zeros(len(powers), dtype=float)
    ends_arr = np.zeros(len(powers), dtype=float)
    for i, v in starts:
        starts_arr[i] = v
    for i, v in ends:
        ends_arr[i] = v
    return FFTReductionOverTime(powers, starts_arr, ends_arr, fftsize, samplerate=samplerate, start_freq=start, end_freq=end)


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
simple_serial_spectral_power_fft_reduce = functools.partial(simple_fft_reduce, serial_spectral_power_fft_reduce)
simple_parallel_spectral_power_fft_reduce = functools.partial(simple_fft_reduce, parallel_spectral_power_fft_reduce)


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


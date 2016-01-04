#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for filter visualization
"""
from scipy import signal
import numpy as np
import numbers


class NotComputedException(Exception):
    "The filter has not been computed yet"
    pass


class FilterUnstableError(Exception):
    """The generated filter is numerically unstable and must not be used"""
    pass


class SignalFilter(object):
    """
    High-level abstraction of a digital signal filter
    """
    def __init__(self, fs, freqs):
        self.nyq = fs * 0.5
        self.fs = fs
        self.b = None
        self.a = None
        self.freqs = self._filtfreq(freqs)

    def _filtfreq(self, f):
        """
        Normalize frequencies with respect to the sampling rate
        """
        if isinstance(f, numbers.Number):
            return f / (0.5 * self.fs)
        else:
            return [f[0] / self.nyq, f[1] / self.nyq]

    def is_stable(self):
        """
        Check if the filter is numerically stable.
        Based on PMcPherson's answer at
        https://github.com/scipy/scipy/issues/2980
        """
        if self.a is None:
            raise NotComputedException()
        return not np.any(np.abs(np.roots(self.a)) > 1.0)

    def iir(self, order, btype="bandpass", ftype="butter", rp=0.01, rs=100.0):
        """
        Generate filter coefficients for an arbitrary IIR filter
        """
        self.b, self.a = signal.iirfilter(order, self.freqs, btype=btype,
                                          ftype=ftype, rp=rp, rs=rs)
        if not self.is_stable():
            self.a = self.b = None
            raise FilterUnstableError()

    def frequency_response(self, n=10000):
        """
        Generate a filter frequency response from a set of filter taps.
        Returns plottable (x, y) with respect to an actual sampling rate
        """
        w, h = signal.freqz(self.b, self.a, worN=n)
        return (0.5 * self.fs * w / np.pi, np.abs(h))

    def __call__(self, d):
        return signal.filtfilt(self.b, self.a, d)

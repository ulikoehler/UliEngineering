#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A high-level API for digital filters:

Features include:
    - Automatic detection of numerical instability
    - Single-line filter generation and application
    - Supports lowpass, highpass, bandpass and bandstop filter types
    - Supports any filter characteristic available in scipy
    - Lets you enter the corner frequencies as Hz,
    - Direct frequency response generation
    - Filters implent __call__ for direct filtfilt application
    - Chainable filter to deal with complex characteristics or unstable high-order filters
"""
from scipy import signal
import numpy as np
import numbers
from toolz import functoolz

__all__ = ["NotComputedException", "FilterUnstableError", "SignalFilter",
           "ChainedFilter"]


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

        Returns the current instance so it can be chained inline
        """
        self.b, self.a = signal.iirfilter(order, self.freqs, btype=btype,
                                          ftype=ftype, rp=rp, rs=rs)
        if not self.is_stable():
            self.a = self.b = None
            raise FilterUnstableError()
        return self

    def frequency_response(self, n=10000):
        """
        Generate a filter frequency response from a set of filter taps.
        Returns plottable (x, y) with respect to an actual sampling rate
        """
        w, h = signal.freqz(self.b, self.a, worN=n)
        return (0.5 * self.fs * w / np.pi, np.abs(h))

    def __call__(self, d):
        if self.a is None:
            raise NotComputedException()
        return signal.filtfilt(self.b, self.a, d)

class ChainedFilter(object):
    """
    Chained filter object that applies a number of filters in series.
    This can be used to deal with numerically unstable filters.

    filtfilt is used to avoid phase issues by repeated application.

    Frequency response plotting is currently unsupported.
    """
    def __init__(self, filters, repeat=1):
        "The first filter in the filters list is applied first"
        if isinstance(filters, SignalFilter):
            filters = [filters]
        self.filters = filters
        self.unique_filters = filters  # Will not contain repeat duplicates
        if repeat > 1:
            self.filters *= repeat

    def __call__(self, d):
        return functoolz.pipe(d, *self.filters)

    def is_stable(self):
        return all([f.is_stable() for f in self.filters])

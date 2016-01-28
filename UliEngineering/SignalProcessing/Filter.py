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
    - ChainedFilter to deal with complex characteristics or unstable high-order filters
    - SumFilter that add the components of multiple individual filters to easily combine multiple
      bandpass filters
    - SumFilter and ChainedFilter are arbitrarily combinable
    - Intuitive, readable error messages for non-mathematicians
"""
from UliEngineering.EngineerIO import autoNormalizeEngineerInputNoUnitRaise
from scipy import signal
import numpy as np
import numbers
import collections
from toolz import functoolz

__all__ = ["NotComputedException", "FilterUnstableError", "SignalFilter",
           "ChainedFilter", "SumFilter"]


class NotComputedException(Exception):
    "The filter has not been computed yet"
    pass


class FilterUnstableError(Exception):
    """The generated filter is numerically unstable and must not be used"""
    pass


class SignalFilter(object):
    """
    High-level abstraction of a digital signal filter.
    """
    def __init__(self, fs, freqs, btype="lowpass"):
        """
        Initialize a new filter

        Keyword arguments:
            fs: The sampling rate
            freqs: The frequency (for lopass/hipass) or a list of two frequencies
        """
        self.nyq = fs * 0.5
        self.fs = autoNormalizeEngineerInputNoUnitRaise(fs)
        self.b = None
        self.a = None
        # Normalize freqs: Allow [1.0] instead of 1.0
        if freqs is None:
            raise ValueError("Critical frequencies may not be none")
        if isinstance(freqs, collections.Iterable) and not isinstance(freqs, str):
            if len(freqs) == 1:
                freqs = freqs[0]
            elif len(freqs) == 0:
                raise ValueError("Empty frequency list")
            elif isinstance(freqs[0], str):
                freqs = [autoNormalizeEngineerInputNoUnitRaise(f) for f in freqs]
        # Allow "4.5 kHz" etc
        if isinstance(freqs, str):
            __freqs_orig = freqs
            freqs = autoNormalizeEngineerInputNoUnitRaise(freqs)
            if freqs is None:
                raise ValueError("Can'")
        self.freqs = self._filtfreq(freqs)
        # Check & store pass type
        if btype == "lowpass" or btype == "highpass":
            if not isinstance(freqs, numbers.Number):
                raise ValueError("Pass-type {0} requires a single critical frequency, not {1}".format(btype, freqs))
        elif btype == "bandpass" or btype == "bandstop":
            if isinstance(freqs, numbers.Number) or len(freqs) != 2:
                raise ValueError("Pass-type {0} requires a two critical frequencies, not {1}".format(btype, freqs))
        else:
            raise ValueError("Invalid pass type '{0}': Use lowpass, highpass, bandpass or bandstop!".format(btype))
        self.btype = btype

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

    def iir(self, order, ftype="butter", rp=0.01, rs=100.0):
        """
        Generate filter coefficients for an arbitrary IIR filter

        Returns the current instance so it can be chained inline
        """
        self.b, self.a = signal.iirfilter(order, self.freqs, btype=self.btype,
                                          ftype=ftype, rp=rp, rs=rs)
        if not self.is_stable():
            self.a = self.b = None
            raise FilterUnstableError("The filter is numerically unstable. Use a lower order or a wider frequency range. You can use ChainedFilter to chain multiple filters of lower order to avoid this issue.")
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

    Filters can be added to the end of the chain via +=
    """
    def __init__(self, filters, repeat=1):
        "The first filter in the filters list is applied first"
        if isinstance(filters, SignalFilter):
            filters = [filters]
        self.filters = filters
        if repeat > 1:
            self.filters *= repeat

    def __iadd__(self, f):
        "Add a filter to the end of the chain"
        self.filters.append(f)
        return self

    def __len__(self):
        return len(self.filters)

    def __call__(self, d):
        return functoolz.pipe(d, *self.filters)

    def frequency_response(self, n=10000):
        if not self.filters:
            raise NotComputedException("Filter list is empty")
        fx, _ = self.filters[0].frequency_response(n)
        fy = np.product(np.asarray([f.frequency_response(n)[1] for f in self.filters]), axis=0)
        return fx, fy

    def is_stable(self):
        # Performance not considered important here. User will usually call this once
        return all(f.is_stable() for f in self.filters)


class SumFilter(ChainedFilter):
    """
    Chained filter object that applies a number of filters and sums the results.
    This can be used to combine multiple bandpass filters for multiple passbands.

    Filters can be added via +=
    """
    def __init__(self, filters):
        "The first filter in the filters list is applied first"
        if isinstance(filters, SignalFilter):
            filters = [filters]
        self.filters = filters

    def __call__(self, d):
        return sum(filt(d) for filt in self.filters)
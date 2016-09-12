#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A python script to normalize a wide variety of value notations
from electronics engineering.

Examples of valid notations include:
    1,234.56kΩ
    1k234
    1k234Ω
    1,234.56Ω
    4µA
    4e6A
    4e6nA

Usage example:
    >>> print(normalizeEngineerInput("1µ234 Ω"))
    (1.234e-6, 'Ω')

Originally published at techoverflow.net.
"""
import re
import math
import itertools
import functools
from collections import namedtuple
import numpy as np

__all__ = ["Quantity", "UnannotatedReturnValueError", "isValidSuffix",
           "getSuffixMultiplier", "normalizeCommaToPoint",
           "splitSuffixSeparator", "normalizeEngineerInput",
           "formatValue", "autoNormalizeEngineerInput",
           "autoNormalizeEngineerInputNoUnit",
           "autoNormalizeEngineerInputNoUnitRaise", "autoFormat",
           "auto_suffix_1d", "split_unit"]

Quantity = namedtuple("Quantity", ["unit"])

class UnannotatedReturnValueError(Exception):
    pass

# Suffices handled by the library
_default_suffices = [["y"], ["z"], ["a"], ["f"], ["p"], ["n"], ["µ", "u"], ["m"], [],
              ["k"], ["M"], ["G"], ["T"], ["E"], ["Z"], ["Y"]]
_default_1st_suffix_exp = -24  # The exponential multiplier for the first suffix

# Valid unit designators. Ensure no SI suffix is added here
_default_units = frozenset(["F", "A", "Ω", "W", "H", "C", "F", "K", "Hz", "V"])

class EngineerIO(object):
    def __init__(self, units=_default_units,
                 unit_prefixes="Δ°"
                 suffices=_default_suffices,
                 first_suffix_exp=_default_1st_suffix_exp):
        """
        Initialize a new EngineerIO instance with default or custom suffix

        Parameters:
        -----------
        units : iterable of strings
            An iterable of valid units (1-char or 2-char)
        unit_prefixes : string
            A list of prefixes that are silently ignored.
            Constraint: unitPrefixes ∩ suffices == ∅
        suffices : list of lists of unit strings
            For each SI exponent, a list of valid suffix strings for each exponent.
            Each successive element in the list is 1e3 from the previous one.
            For generating strings from numbers, the first suffix in each nested list is preferred
        first_suffix_exp : int
            The decimal exponent of the first suffix in the suffix list.
        """
        self.units = set(units)
        self.suffices = suffices
        self.first_suffix_exp = first_suffix_exp
        # Compute inverse suffix map
        self.suffix_map = {}  # Key: exp // 3, Value: suffix
        current_exp = first_suffix_exp
        for current_suffices in suffices:
            # Use only first suffix
            current_suffix = current_suffices[0]
            self.suffix_map[current_exp // 3] = current_suffix
            # Next exponent
            current_exp += 3
        # Compute min/max SI value
        self.suffix_map_min = min(self.suffix_map.keys())
        self.suffix_map_max = max(self.suffix_map.keys())
        # Compute list of all suffixes
        self.all_suffixes = set(itertools.chain(*self.suffices))


def isValidSuffix(suffix):
    """
    Check if a given string is valid when used in getSuffixMultiplier()
    """
    if not suffix:  # 0 multiplier
        return True
    for sfx in itertools.chain(*siSuffices):
        if suffix == sfx:
            return True
    return False

def getSuffixMultiplier(suffix):
    """
    For a given character, get either the multiplier or None if not found.
    The multiplier is returned as base-10 exponent integral. This avoids
    IEEE754 inaccuracies.
    """
    if not suffix:
        return 0
    multiplier = siSuffixMult
    for suffixList in siSuffices:
        for siSuffix in suffixList:
            if siSuffix == suffix:
                return multiplier
        multiplier += 3
    return None

def normalizeCommaToPoint(s):
    """
    Normalize comma to point for float conversion.
    Correctly handles thousands separators.

    Note that cases like "1,234" are undecidable between
    "1234" and "1.234". They are treated as "1.234".

    Only points and commata are potentially modified.
    Other characters and digits are not handled.
    """
    foundCommaFirst = False
    commaIdx = s.find(",")
    pointIdx = s.find(".")
    foundComma = commaIdx is not None
    foundPoint = pointIdx is not None
    foundCommaFirst = commaIdx < pointIdx if foundComma and foundPoint else None
    # Found nothing or only point -> no modification required
    if not (foundComma or foundPoint) or (foundPoint and not foundComma):
        return s
    # Only comma -> replace and exit
    if foundComma and not foundPoint:
        return s.replace(",", ".")
    # foundComma and foundPoint
    if foundCommaFirst:  # Comma likely used as thousands separators
        # Just remove commata
        return s.replace(",", "")
    else:  # Point used as thousands separator
        return s.replace(".", "").replace(",", ".")

def split_unit(s):
    """
    Split a string into (remainder, unit).
    Only units in the units set are recognized
    unit may be "None" if no unit is recognized
    """
    # Handle 2-character units (e.g. 'Hz'), then 1-character unity (e.g. 'V')
    if s[-2:] in units: # Will also handle unit-only 1-char strings
        s, unit = s[:-2], s[-2:]
    elif s[-1] in units:  # Handle 1-char units
        s, unit = s[:-1], s[-1]
    else: # No unit
        s, unit = s, ''
    # Remove unit prefix, if any (e.g. degrees symbol, delta symbol)
    s = s.strip().rstrip(unitPrefixes).strip()
    return s, unit

def splitSuffixSeparator(s):
    """
    Separate a string into a 3-tuple (number, suffix, unit).
    returns None if the string could not be parsed.

    The tuple will never contain None but empty strings if some
    element is not present. The number must be present for the string
    to be considered valid.

    Units always need to be a suffix. Instead
    Thousands separators or commata instead of points may be used
    (see normalizeCommaToPoint documentation).

    Thousands separators and suffix-as-decimal-separators may NOT
    be mixed. Whitespace is removed automatically.
    """
    if not s:
        return None
    s = normalizeCommaToPoint(s).replace(" ", "")
    # Ensure we have at least one character
    if not s:
        return None
    s, unit = split_unit(s)
    # The string with possibly the unit removed must be non-empty
    if not s:
        return None
    # Try to find SI suffix at the end or in the middle
    if isValidSuffix(s[-1]): # At the end
        suffix = s[-1]
        s = s[:-1]
    else:  # Try to find unit anywhere
        isSuffixList = [isValidSuffix(ch) for ch in s]
        # Ensure only ONE unit occurs in the string
        suffixCount = sum(isSuffixList)
        if suffixCount > 1:
            return None
        elif suffixCount == 0:
            suffix = ""
        else:  # suffixCount == 1 --> correct
            # Suffix-as-decimal-separator --> no other decimal separator
            if "." in s:  # Commas already handled by normalizeCommaToPoint
                return None
            suffixIndex = isSuffixList.index(True)
            # Suffix must NOT be first character
            if suffixIndex == 0:
                return None
            suffix = s[suffixIndex]
            s = s.replace(suffix, ".")
    # Handle unit prefix (if any). Not allowable if no unit is present
    if not s:
        return None
    if unit and s[-1] in unitPrefixes:
        s = s[:-1]
    # Final check: Is there any number left and is it valid?
    if not s:
        return None
    if not all([ch.isdigit() or ch in [".", "-", "e"] for ch in s]):
        return None
    return (s, suffix, unit)

def normalizeEngineerInput(s):
    """
    Converts an engineer's input of a wide variety of formats to a numeric
    value.

    Returns a pair (number, unit) or None if the conversion could not be performed.

    See splitSuffixSeparator() for further details on supported formats
    """
    res = splitSuffixSeparator(s)
    if res is None: return None
    (num, suffix, unit) = res
    val = float(num) * (10 ** getSuffixMultiplier(suffix))
    return (val, unit)


def _formatWithSuffix(v, suffix):
    """
    Format a given value with a given suffix.
    This helper function formats the value to 3 visible digits.
    v must be pre-multiplied by the factor implied by the suffix
    """
    if v < 10.0:
        res = "{:.2f}".format(v)
    elif v < 100.0:
        res = "{:.1f}".format(v)
    else:  # Should only happen if v < 1000
        res = str(int(round(v)))
    #Avoid appending whitespace if there is no suffix
    if suffix:
        return "{0} {1}".format(res, suffix)
    else:
        return res

def formatValue(v, unit=""):
    """
    Format v using SI suffices with optional units.
    Produces a string with 3 visible digits.
    """
    #Suffix map is indexed by one third of the decadic logarithm.
    exp = 0 if v == 0.0 else math.log(abs(v), 10.0)
    suffixMapIdx = int(math.floor(exp / 3.0))
    #Ensure we're in range
    if not siSuffixMapMin < suffixMapIdx < siSuffixMapMax:
        return None
    #Pre-multiply the value
    v = v * (10.0 ** -(suffixMapIdx * 3))
    #Delegate the rest of the task to the helper
    return _formatWithSuffix(v, siSuffixMap[suffixMapIdx] + unit)

def autoNormalizeEngineerInput(v, encoding="utf-8"):
    "Return v, None if v is not a string or normalizeEngineerInput(v) else"
    if isinstance(v, bytes):
        v = v.decode(encoding)
    if isinstance(v, str):
        return normalizeEngineerInput(v)
    return v, ''

def autoNormalizeEngineerInputNoUnit(s):
    """
    Returns the only the value (not the unit) when parsing
    via autoNormalizeEngineerInput(). Returns None on failure.
    """
    v = autoNormalizeEngineerInput(s)
    if v is None:
        return None
    return v[0]

def autoNormalizeEngineerInputNoUnitRaise(s):
    """
    Returns the only the value (not the unit) when parsing
    via autoNormalizeEngineerInput(). Raises ValueError() on failure
    """
    v = autoNormalizeEngineerInput(s)
    if v is None:
        raise ValueError("Could not parse value from '{0}'".format(s))
    return v[0]


def autoFormat(fn, *args, **kwargs):
    """
    Auto-format a value by leveraging function annotations.
    The function's return value is expected to a be annotated with a Quantity() value.
    """
    if not callable(fn):
        raise ValueError("fn must be callable")
    # Access innermost function inside possibly nested partials
    annotatedFN = fn
    while isinstance(annotatedFN, functools.partial):
        annotatedFN = annotatedFN.func
    try:
        qty = annotatedFN.__annotations__["return"]
    except KeyError:
        raise UnannotatedReturnValueError("Function {0} does not have an annotated return value")
    return formatValue(fn(*args, **kwargs), qty.unit)

def auto_suffix_1d(arr):
    """
    Takes an array of arbitrary values and determines
    what is the best suffix (e.g. M, m, n, f) to represent
    as many values as possible with as few powers of 10 as possible.

    Returns a tuple (factor, suffix) where the factor is a floating-point
    value to multiply the array with to obtain value with "suffix" suffix.
    """
    # Compute logarithmic magnitudes of data
    arr_log = np.log10(np.abs(arr))
    arr_log[np.isinf(arr_log)] = 0  # log(0) == inf
    log_mean = arr_log.mean()
    # Generate score matrix
    suffix_idx = int(round(log_mean / 3))
    # Ensure we're in range
    suffix_idx = max(siSuffixMapMin, suffix_idx)
    suffix_idx = min(siSuffixMapMax, suffix_idx)
    # Pre-multiply the value
    multiplier = 10.0 ** -(suffix_idx * 3)
    return multiplier, siSuffixMap[suffix_idx]

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
import sys
import six

__author__ = "Uli Koehler"
__license__ = "CC0 1.0 Universal"
__version__ = "1.0"

# Suffices handled by the library
siSuffices = [["f"], ["p"], ["n"], ["u", "µ"], ["m"], [],
              ["k"], ["M"], ["G"], ["T"]]
siSuffixMult = -15  # The multiplier for the first suffix
siSuffixMap = {
    -5: "f", -4: "p", -3: "n", -2: "µ", -1: "m",
    0: "", 1: "k", 2: "M", 3: "G", 4: "T", 5: "E"
}

# Valid unit designators. Ensure no SI suffix is added here
units = frozenset(["F", "A", "Ω", "W", "H", "C", "F", "K", "Hz", "V"])

# Allowable Unit prefixes
# Constraint: unitPrefixes ∩ siSuffices == ∅
unitPrefixes = frozenset(["Δ", "°"])

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


def isValidSuffix(suffix):
    """Check if the given character(s) represent a valid suffix"""
    return getSuffixMultiplier(suffix) is not None


def normalizeCommaToPoint(s):
    """
    Normalize comma to point for float conversion.
    Correctly handles thousands separators.

    Note that cases like "1,234" are undecidable between
    "1234" and "1.234". They are treated as "1.234".

    Only points and commata are potentially modified.
    Other characters and digits are not handled.
    """
    foundComma = False
    foundPoint = False
    foundCommaFirst = False
    for ch in s:
        if ch == ",":
            # Set flag if this is the first comma and no
            # point has been encountered so far
            if not (foundPoint or foundComma):
                foundCommaFirst = True
            foundComma = True
        elif ch == ".":
            foundPoint = True
    # Found nothing or only point -> no modification required
    if not (foundComma or foundPoint) or (foundPoint and not foundComma):
        return s
    # Only comma -> replace and exit339.photobucket.com/user/ulitronic/library
    if foundComma and not foundPoint:
        return s.replace(",", ".")
    # foundComma and foundPoint
    if foundCommaFirst:  # Comma likely used as thousands separators
        # Just remove commata
        return s.replace(",", "")
    else:  # Point used as thousands separator
        return s.replace(".", "").replace(",", ".")

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
    # Handle 2-character units (MUST be a suffix)
    if len(s) > 2 and s[-2:] in units:
        unit = s[-2:]
        s = s[:-2]
    else:  # Handle 1-char units
        # If this is executed, the unit MUST be a suffix and 1 char only
        unit = s[-1] if s[-1] in units else ""
        if unit:  # Strip unit from string
            s = s[:-1]
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
        suffixCount = isSuffixList.count(True)
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
    if v < 10:
        res = "%.2f" % v
    elif v < 100:
        res = "%.1f" % v
    else:  # Should only happen if v < 1000
        res = "%d" % int(v)
    #Avoid appending whitespace if there is no suffix
    if suffix:
        return "%s %s" % (res, suffix)
    else:
        return res

def formatValue(v, unit=""):
    """
    Format v using SI suffices with optional units.
    Produces a string with 3 visible digits.
    """
    #Suffix map is indexed by one third of the decadic logarithm.
    exp = 0 if v == 0.0 else math.log(v, 10.0)
    suffixMapIdx = int(math.floor(exp / 3.0))
    #Ensure we're in range
    if suffixMapIdx < -5:
        suffixMapIdx = -5
    elif suffixMapIdx > 5:
        suffixMapIdx = 5
    #Pre-multiply the value
    v = v * (10.0 ** -(suffixMapIdx * 3))
    #Delegate the rest of the task to the helper
    return _formatWithSuffix(v, siSuffixMap[suffixMapIdx] + unit)

def normalizeEngineerInputIfStr(v):
    "Return v, None if v is not a string or normalizeEngineerInput(v) else"
    if isinstance(v, six.binary_type):
        v = v.decode("utf-8")
    if isinstance(v, six.text_type):
        return normalizeEngineerInput(v)
    return v, ''

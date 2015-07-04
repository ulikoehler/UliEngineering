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

Originally published at techoverflow.net.
"""
import re

__author__ = "Uli Koehler"
__license__ = "CC0 1.0 Universal"
__version__ = "1.0"

# Suffices handled by the library
siSuffices = [["f"], ["p"], ["n"], ["u", "µ"], ["m"], [],
              ["k"], ["M"], ["G"], ["T"]]
siSuffixMult = -15  # The multiplier for the first suffix

# Valid unit designators. Ensure no suffix is added here
units = ["F", "A", "Ω", "W", "H"]

def getSuffixMultiplier(suffix):
    """
    For a given character, get either the multiplier or None if not found.
    The multiplier is returned as base-10 exponent integral. This avoids
    IEEE754 inaccuracies.

    >>> getSuffixMultiplier("f")
    -15
    >>> getSuffixMultiplier("k")
    3
    >>> getSuffixMultiplier("u")
    -6
    >>> getSuffixMultiplier("µ")
    -6
    >>> getSuffixMultiplier("T")
    12
    >>> getSuffixMultiplier("")
    0
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
    """
    Check if the given character is a valid suffix

    >>> isValidSuffix("f")
    True
    >>> isValidSuffix("k")
    True
    >>> isValidSuffix("T")
    True
    >>> isValidSuffix("µ")
    True
    >>> isValidSuffix("B")
    False
    """
    return getSuffixMultiplier(suffix) is not None


def normalizeCommaToPoint(s):
    """
    Normalize comma to point for float conversion.
    Correctly handles thousands separators.

    Note that cases like "1,234" are undecidable between
    "1234" and "1.234". They are treated as "1.234".

    Only points and commata are potentially modified.
    Other characters and digits are not handled.

    >>> normalizeCommaToPoint("1234")
    '1234'
    >>> normalizeCommaToPoint("123.4")
    '123.4'
    >>> normalizeCommaToPoint("123,4")
    '123.4'
    >>> normalizeCommaToPoint("1,234.5")
    '1234.5'
    >>> normalizeCommaToPoint("1.234,5")
    '1234.5'
    >>> normalizeCommaToPoint("1.234,5k")
    '1234.5k'
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

    >>> splitSuffixSeparator("1234")
    ('1234', '', '')
    >>> splitSuffixSeparator("1234k")
    ('1234', 'k', '')
    >>> splitSuffixSeparator("1234kΩ")
    ('1234', 'k', 'Ω')
    >>> splitSuffixSeparator("1.234kΩ")
    ('1.234', 'k', 'Ω')
    >>> splitSuffixSeparator("1,234kΩ")
    ('1.234', 'k', 'Ω')
    >>> splitSuffixSeparator("1,234.56kΩ")
    ('1234.56', 'k', 'Ω')
    >>> splitSuffixSeparator("1k234")
    ('1.234', 'k', '')
    >>> splitSuffixSeparator("1k234Ω")
    ('1.234', 'k', 'Ω')
    >>> splitSuffixSeparator("1,234.56Ω")
    ('1234.56', '', 'Ω')
    >>> splitSuffixSeparator("1A")
    ('1', '', 'A')
    >>> splitSuffixSeparator("1")
    ('1', '', '')
    >>> splitSuffixSeparator("1k234 Ω")
    ('1.234', 'k', 'Ω')
    >>> splitSuffixSeparator("1")
    ('1', '', '')
    >>> splitSuffixSeparator("-1,234.56kΩ")
    ('-1234.56', 'k', 'Ω')
    >>> splitSuffixSeparator("-1e3kΩ")
    ('-1e3', 'k', 'Ω')
    >>> splitSuffixSeparator("1e-3kΩ")
    ('1e-3', 'k', 'Ω')
    >>> splitSuffixSeparator("-4e6nA")
    ('-4e6', 'n', 'A')
    >>> splitSuffixSeparator("1,234.56kfA")
    >>> splitSuffixSeparator("1.23k45A")
    >>> splitSuffixSeparator("")
    >>> splitSuffixSeparator("1,234k56Ω")
    >>> splitSuffixSeparator("foobar")
    >>> splitSuffixSeparator(None)
    """
    if not s:
        return None
    s = normalizeCommaToPoint(s).replace(" ", "")
    # Ensure we have at least one character
    if not s:
        return None
    # If there is a unit, it MUST be a suffix and 1 char only
    unit = s[-1] if s[-1] in units else ""
    if unit:  # Remove unit from current string
        s = s[:-1]
        # The stripped string must be non-empty
        if not s:
            return None
    # Try to find SI suffix at the end or in the middle
    if isValidSuffix(s[-1]):
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

    Returns a pair (numb)

    See splitSuffixSeparator() for further details on supported formats
    """
    (num, suffix, unit) = splitSuffixSeparator(s)
    val = float(num) * (10 ** getSuffixMultiplier(suffix))
    return (val, unit)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Usage example
    print(normalizeEngineerInput("1µ234 Ω"))
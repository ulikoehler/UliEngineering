#!/usr/bin/env python3
"""
String utilities and algorithms
"""

__all__ = ["split_nth", "suffix_list", "partition_at_numeric_to_nonnumeric_boundary"]

from typing import List, Tuple
import re

def split_nth(s, delimiter=",", nth=1):
    """
    Like s.split(delimiter), but only returns the nth string of split's return array.
    Other strings or the split list itself are not generated.

    Using this function is ONLY recommended (because it's ONLY faster)
    if the string contains MANY delimiters (multiple hundreds).
    Else, use s.split(delimiter)[n - 1]

    Throws ValueError if the nth delimiter has not been found.
    """
    if nth <= 0:
        raise ValueError("Invalid nth parameter: Must be >= 0 but value is {0}".format(nth))
    startidx = 0
    # Startidx is 0 if we want the first field
    if nth > 1:
        for _ in range(nth - 1):
            startidx = s.index(delimiter, startidx + 1)
        startidx += 1  # Do not include the delimiter
    # Determine end index
    endidx = s.find(delimiter, startidx)
    if endidx == -1:  # Not found -> the last part of the string
        endidx = None  # Take rest of string
    return s[startidx:endidx]

def suffix_list(s: str) -> List[str]:
    """
    Return all suffixes for a string, including the string itself,
    in order of ascending length.

    Example: "foobar" => ['r', 'ar', 'bar', 'obar', 'oobar', 'foobar']
    """
    return [s[-i:] for i in range(1, len(s) + 1)]

_numeric_to_nonnumeric_boundary_regex = re.compile(r"([-\.\d]+)([^\d\.]+)")

def partition_at_numeric_to_nonnumeric_boundary(s: str) -> Tuple[str, str]:
    """
    Partition a string at the first numeric->non-numeric boundary.
    Returns a tuple of two strings.

    Examples:
        * "foo.123bar" => ("foo.123", "bar")
        * "123s" => ("123", "s")
        * "123" => ("123", "")
        * "123.456km" => ("123.456", "km")
        * "foo" => ("foo", "")
        * "foo1bar" => ("foo1", "bar")
    """
    m = _numeric_to_nonnumeric_boundary_regex.search(s)
    if m is None:
        # No such boundary found
        return s, ""
    return s[:m.span(1)[1]], s[m.span(1)[1]:]

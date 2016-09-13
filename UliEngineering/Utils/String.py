#!/usr/bin/env python3
"""
String utilities and algorithms
"""

__all__ = ["split_nth"]


def split_nth(s, delimiter=",", nth=1):
    """
    Like s.split(delimiter), but only returns the nth string of split's return array.
    Other strings or the split list itself are not generated, therefore using this function
    might be more efficient in many usecases where only
    Does not support regex delimiters.

    Throws ValueError if the nth delimiter has not been found
    """
    if nth <= 0:
        raise ValueError("Invalid nth parameter: Must be >= 0 but value is {0}".format(nth))
    startidx = 0
    # Startidx is 0 if we want the first field
    if nth > 1:
        for i in range(nth - 1):
            startidx = s.index(delimiter, startidx + 1)
        startidx += 1 # Do not include the delimiter
    # Determine end index
    endidx = s.find(delimiter, startidx)
    if endidx == -1: # Not found -> the last part of the string
        endidx = None # Take rest of string
    return s[startidx:endidx]


def split_nth_x(s, delimiter=",", nth=1):
    # Find start index
    if nth <= 0:
        raise ValueError("Invalid nth parameter: Must be >= 0 but value is {0}".format(nth))
    if nth == 1:
        startidx = 0
    else:
        nth_tmp = nth # This one is counted down
        for startidx in range(len(s)):
            if s[startidx] == delimiter:
                nth_tmp -= 1
                if nth_tmp <= 1:
                    break
        # Possibly we did not find enough delimiters
        if startidx == len(s) - 1 and s[-1] != delimiter:
            raise ValueError("nth value out of range: Not enough '{0}'' delimiters in '{1}'".format(delimiter, s))
    # Find end index
    endidx = None
    for endidx in range(startidx + 1, len(s)):
        if s[endidx] == delimiter:
            break
    # If we did not find the end delimiter, we must avoid skipping the last character,
    # But a delimiter must not be assumed as last character
    if endidx == len(s) - 1 and s[-1] != delimiter:
        endidx = None
    return s[startidx + 1 if nth > 1 else 0:endidx]

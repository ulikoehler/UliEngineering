#!/usr/bin/env python3

__all__ = ["parse_int_or_float", "try_parse_int_or_float"]

def parse_int_or_float(s):
    """
    Try to parse the given string
    as int, and if that fail, as float.

    If the parsing as float fails, raises ValueError.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)


def try_parse_int_or_float(s):
    """
    Try to parse the given string
    as int, and if that fail, as float.

    If the parsing as float fails, returns the string.
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
        
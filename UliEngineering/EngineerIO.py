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
    >>> print(normalize_engineer_notation("1µ234 Ω"))
    (1.234e-6, 'Ω')

Originally published at techoverflow.net.
"""
import math
import itertools
from toolz import functoolz
import numpy as np
from .Units import *
from .Utils.String import suffix_list

__all__ = ["normalize_interpunctation", "EngineerIO",
           "auto_format", "normalize_numeric", "format_value", "auto_print",
           "normalize_engineer_notation", "normalize_engineer_notation_safe"]


def _default_suffices():
    """
    The default first suffix list with -24 1st exp
    """
    return [["y"], ["z"], ["a"], ["f"], ["p"], ["n"], ["µ", "μ", "u"], ["m"], [],
            ["k"], ["M"], ["G"], ["T"], ["E"], ["Z"], ["Y"]]

def _length_units(include_m=False):
    """
    All known length units.
    "m" is not included by default due to 
    See also Length.py
    """
    units = set([
        # Length
        'Å', 'Angstrom', 'angstrom',
        'meters', 'meter',
        'mil', 'in', '\"', 'inch', 'inches',
        'foot', 'feet', 'ft', 'yd', 'yard', 'mile',
        'miles', 'pt', 'point', 'points', 'au', 'AU', 'AUs',
        'ly', 'light year', 'lightyear', 'light years', 'lightyears',
        'nautical mile', 'nautical miles', 'pc', 'parsec', 'parsecs'
    ])
    if include_m:
        units.add("m")
    return units

def _default_units(include_m=False):
    return set([
        'F', 'A', 'Ω', 'W', 'H', 'C', 'K', 'Hz', 'V', 'J', 'S',
        # Time
        's', 'h', 'min',
        # Fraction
        'ppm', 'ppb', '%',
        # Composite units
        '°C/W', 'C/W'
    ]).union(_length_units(include_m=include_m))

# Valid unit designators. Ensure no SI suffix is added here
_numeric_allowed = set("+0123456789-e.")

class EngineerIO(object):
    instance = None
    """
    Default instance, used for global functions. Initialized on first use

    Note: ppm, ppb and % are special 'units' that are handled separately.
    """
    def __init__(self, units=_default_units(),
                 unit_prefixes="Δ°",
                 suffices=_default_suffices(),
                 first_suffix_exp=-24):
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
            The list must not contain the empty string but shall be empty if no suffix shall be used
        first_suffix_exp : int
            The decimal exponent of the first suffix in the suffix list.
        """
        self.units = set(units)
        self.suffices = suffices
        self.first_suffix_exp = first_suffix_exp
        # Unit prefixes will only be used in strip, so we can strip spaces in one go.
        self.strippable = unit_prefixes + " \t\n"
        # Compute maps
        self.all_suffixes = set(itertools.chain(*self.suffices))
        self._recompute_suffix_maps()

    def _recompute_suffix_maps(self):
        """
        Recompute the exponent -> suffix map and
        the suffix -> exponent map
        """
        # Compute inverse suffix map
        self.exp_suffix_map = {}  # Key: exp // 3, Value: suffix
        self.suffix_exp_map = {}  # Key: suffix, value: exponent
        current_exp = self.first_suffix_exp
        # Iterate over first suffices in each list
        for current_suffices in self.suffices:
            if not current_suffices: # No suffix to be used
                # Do not self suffix exp map
                self.exp_suffix_map[current_exp // 3] = ""
                current_exp += 3
                continue
            # Compute exponent -> suffix (only first suffix)
            self.exp_suffix_map[current_exp // 3] = current_suffices[0]
            # Compute suffix -> exponent
            for current_suffix in current_suffices:
                self.suffix_exp_map[current_suffix] = current_exp
            current_exp += 3
        # Compute min/max SI value
        self.exp_map_min = min(self.exp_suffix_map.keys())
        self.exp_map_max = max(self.exp_suffix_map.keys())

    def split_input(self, s):
        """
        Separate a string into a 3-tuple (number, suffix, unit).
        returns None if the string could not be parsed.

        The tuple will never contain None but empty strings if some
        element is not present. The number must be present for the string
        to be considered valid.

        Units always need to be a suffix. Instead
        Thousands separators or commata instead of points may be used
        (see normalize_interpunctation documentation).

        Thousands separators and suffix-as-decimal-separators may NOT
        be mixed. Whitespace is removed automatically.
        """
        # Remove thousands separator & ensure dot is used
        s = normalize_interpunctation(s)
        s, unit = self.split_unit(s) # Remove unit
        s = s.replace(" ", "")
        # Check string
        if not s:
            raise ValueError("Can't split empty string")
        # Try to find SI suffix at the end or in the middle
        if s[-1] in self.all_suffixes:
            s, suffix = s[:-1], s[-1]
        else:  # Try to find unit anywhere
            isSuffixList = [ch in self.all_suffixes for ch in s]
            # Ensure only ONE unit occurs in the string
            suffixCount = sum(isSuffixList)
            if suffixCount > 1:
                raise ValueError("More than one SI suffix in the string")
            elif suffixCount == 0:
                suffix = ""
            else:  # suffixCount == 1 => correct
                # Suffix-as-decimal-separator --> there must be no other decimal separator
                if "." in s:  # Commata already handled by normalize_interpunctation
                    raise ValueError("Suffix as decimal separator, but dot is also in string: {}".format(s))
                suffixIndex = isSuffixList.index(True)
                # Suffix must NOT be first character
                if suffixIndex == 0:
                    raise ValueError("Suffix in '{}' must not be the first char".format(s))
                suffix = s[suffixIndex]
                s = s.replace(suffix, ".")
        # Handle unit prefix (if any). Not allowable if no unit is present
        s = s.strip(self.strippable)
        # Final check: Is there any number left and is it valid?
        if not all((ch in _numeric_allowed for ch in s)):
            raise ValueError("Remainder of string is not purely numeric: {}".format(s))
        return (s, suffix, unit)

    def split_unit(self, s):
        """
        Split a string into (remainder, unit).
        Only units in the units set are recognized
        unit may be '' if no unit is recognized
        """
        # Fallback for strings which are too short
        if len(s) <= 1:
            return s, ""
        # Handle unit suffixes: "ppm"
        # We try to find the longest unit suffix, up to the first digit
        found_unit_suffix = False
        for suffix in suffix_list(s):
            if suffix in self.units:
                # Do not try to find units if encountering the first digit
                if suffix[0].isnumeric():
                    break
                suffix_length = len(suffix)
                value_str, unit = s[:-suffix_length], s[-suffix_length:]
                found_unit_suffix = True
        # Fallback: Try to parse as value + optionally SI postfi
        if not found_unit_suffix: # No unit
            value_str, unit = s, ''
        # Remove unit prefix, if any (e.g. degrees symbol, delta symbol)
        value_str = value_str.rstrip(self.strippable)
        return value_str, unit

    def normalize(self, s, encoding="utf8"):
        """
        Converts an engineer's input of a wide variety of formats to a numeric
        value.

        Returns a pair (number, unit) or None if the conversion could not be performed.

        See splitSuffixSeparator() for further details on supported formats
        """
        # Scalars get returned directly
        if isinstance(s, (int, float, np.generic)):
            return s, ''
        # Make sure it's a decoded string
        if isinstance(s, bytes):
            s = s.decode(encoding)
        # Handle lists / array
        if isinstance(s, (list, tuple, np.ndarray)):
            return [self.normalize(elem) for elem in s]
        # Perform splitting
        (num, suffix, unit) = self.split_input(s.strip())
        mul = (10 ** self.suffix_exp_map[suffix]) if suffix else 1
        # Handle ppm and ppb: They are listed as units
        if unit == '%':
            mul /= 100
            unit = ''
        elif unit == 'ppm':
            mul /= 1e6
            unit = ''
        elif unit == 'ppb':
            mul /= 1e9
            unit = ''
        return (float(num) * mul, unit)

    def safe_normalize(self, s, encoding="utf8"):
        """
        Same as normalize(), but returns None instead of raising
        on error.
        """
        try:
            return self.normalize(s, encoding)
        except ValueError:
            return None

    def format(self, v, unit="", significant_digits=3):
        """
        Format v using SI suffices with optional units.
        Produces a string with 3 visible digits.
        """
        if unit is None:
            unit = ""
        #Suffix map is indexed by one third of the decadic logarithm.
        exp = 0 if v == 0. else math.log(abs(v), 10.)
        suffixMapIdx = int(math.floor(exp / 3.))
        #Ensure we're in range
        if not self.exp_map_min < suffixMapIdx < self.exp_map_max:
            raise ValueError("Value out of range: {}".format(v))
        #Pre-multiply the value
        v = v * (10.0 ** -(suffixMapIdx * 3))
        #Delegate the rest of the task to the helper
        return _format_with_suffix(
            v,
            self.exp_suffix_map[suffixMapIdx] + unit,
            significant_digits=significant_digits
        )

    def auto_suffix_1d(self, arr):
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
        suffix_idx = max(self.exp_map_min, suffix_idx)
        suffix_idx = min(self.exp_map_max, suffix_idx)
        # Pre-multiply the value
        multiplier = 10.0 ** -(suffix_idx * 3)
        return multiplier, self.exp_suffix_map[suffix_idx]

    def auto_format(self, fn, *args, significant_digits=3, **kwargs):
        """
        Auto-format a value by leveraging function annotations.
        For example this can be used to format using UliEngineering Physics quantities
        The function's return value is expected to a be annotated with a Quantity() value.
        """
        unit = find_returned_unit(fn)
        return self.format(fn(*args, **kwargs), unit=unit, significant_digits=significant_digits)

    def auto_print(self, *args, **kwargs):
        print(self.auto_format(*args, **kwargs))

    def normalize_numeric_safe(self, arg):
        """
        Normalize each element of an iterable and retrieve only the numeric value
        (the unit is ignored). Works on iterables and string-likes.

        Use toolz.itertoolz.compact() on the result to remove all None values.

        Returns an ndarray with np.nan (on error) or the numeric value (no unit).

        """
        if arg is None:
            raise ValueError("Can't normalize None")
        # Scalars get returned directly
        if isinstance(arg, (int, float, np.generic)):
            return arg
        # If it's stringlike, apply directly
        if isinstance(arg, str) or isinstance(arg, bytes):
            v = self.safe_normalize(arg)
            return None if v is None else v[0]
        # It's an iterable
        ret = np.zeros(len(arg))
        for i, elem in enumerate(arg):
            v = self.safe_normalize(elem)
            ret[i] = np.nan if v is None else v[0]
        return ret

    def normalize_numeric(self, arg):
        """
        Normalize each element of an iterable and retrieve only the numeric value
        (the unit is ignored). Works on iterables and string-likes.

        Raises if any of the values can't be normalized.

        If the given value is an iterable, a ndarray is returned.
        """
        if arg is None:
            raise ValueError("Can't normalize None")

        # Scalars get returned directly
        if isinstance(arg, (int, float, np.generic)):
            return arg

        # If it's stringlike, apply directly
        if isinstance(arg, (str, bytes)):
            return self.normalize(arg)[0]
        # It's an iterable
        ret = np.zeros(len(arg))
        for i, elem in enumerate(arg):
            ret[i] = self.normalize(elem)[0]
        return ret

# Initialize global instance
EngineerIO.instance = EngineerIO()
EngineerIO.length_instance = EngineerIO(units=_default_units(include_m=True))

__replace_comma_dot = lambda s: s.replace(",", ".")
"""
Map of a transform to apply to the string
during interpunctation normalization,
depending on (commaFound, dotFound, commaFoundFirst).
Must contain every possible variant
"""
_interpunct_transform_map = {
    # Found nothing or only point -> no modification required
    (False, False, False): functoolz.identity,
    (False, False, True): functoolz.identity,
    (False, True, False): functoolz.identity,
    (False, True, True): functoolz.identity,
    # Only comma -> replace and exit
    (True, False, False): __replace_comma_dot,
    (True, False, True): __replace_comma_dot,
    # Below this line: Both comma and dot found
    # Comma first => comma used as thousands separators
    (True, True, True): lambda s: s.replace(",", ""),
    # Dot first => dot used as thousands separator
    (True, True, False): lambda s: s.replace(".", "").replace(",", ".")
}

def normalize_interpunctation(s):
    """
    Normalize comma to point for float conversion.
    Correctly handles thousands separators.

    Note that cases like "1,234" are undecidable between
    "1234" and "1.234". They are treated as "1.234".

    Only points and commata are potentially modified.
    Other characters and digits are not handled.
    """
    commaIdx = s.find(",")
    pointIdx = s.find(".")
    foundComma = commaIdx is not None
    foundPoint = pointIdx is not None
    commaFirst = commaIdx < pointIdx if (foundComma and foundPoint) else None
    # Get the appropriate transform function from the map an run it on s
    return _interpunct_transform_map[(foundComma, foundPoint, commaFirst)](s)


def _format_with_suffix(v, suffix="", significant_digits=3):
    """
    Format a given value with a given suffix.
    This helper function formats the value to 3 visible digits.
    v must be pre-multiplied by the factor implied by the suffix.

    Keyword arguments
    -----------------
    suffix : string
        The suffix to append
    significant_digits : integer
        The number of overall significant digits to show
    """
    if v < 1.0:
        res = ("{:." + str(significant_digits - 0) + "f}").format(v)
    if v < 10.0:
        res = ("{:." + str(significant_digits - 1) + "f}").format(v)
    elif v < 100.0:
        res = ("{:." + str(significant_digits - 2) + "f}").format(v)
    else:  # Should only happen if v < 1000
        res = str(int(round(v)))
    #Avoid appending whitespace if there is no suffix
    return "{} {}".format(res, suffix) if suffix else res

def normalize_engineer_notation(s, encoding="utf8"):
    return EngineerIO.instance.normalize(s)

def format_value(v, unit="", significant_digits=3):
    return EngineerIO.instance.format(v, unit, significant_digits=significant_digits)

def normalize_engineer_notation_safe(v, unit=""):
    return EngineerIO.instance.safe_normalize(v, unit)

def normalize_numeric(v):
    return EngineerIO.instance.normalize_numeric(v)

def normalize(v):
    return EngineerIO.instance.normalize(v)

def auto_format(v, *args, **kwargs):
    return EngineerIO.instance.auto_format(v, *args, **kwargs)

def auto_print(*args, **kwargs):
    return EngineerIO.instance.auto_print(*args, **kwargs)

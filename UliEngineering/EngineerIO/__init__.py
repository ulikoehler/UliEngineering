#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A python script to normalize a wide variety of value notations

Examples of valid notations include:
    1,234.56kΩ
    1k234
    1k234Ω
    1,234.56Ω
    4µA
    4e6A
    4e6nA

Usage example:
    >>> print(normalize_engineer_notation("1µ234 Ω"))
    (1.234e-6, 'Ω')

Originally published at techoverflow.net.
"""
from ast import alias
from collections.abc import Iterable
import math
import re
from typing import Dict, List, Optional, Set
import numpy as np
import functools
import inspect
import scipy as sp
from toolz import functoolz
from dataclasses import dataclass
from ..Exceptions import MultipleUnitPrefixesException, RemainderOfStringContainsNonNumericCharacters, FirstCharacterInStringIsUnitPrefixException

from UliEngineering.Units import InvalidUnitInContextException, UnannotatedReturnValueError
from ..Utils.String import partition_at_numeric_to_nonnumeric_boundary

__all__ = ["normalize_interpunctation", "EngineerIO",
           "auto_format", "normalize_numeric", "format_value", "auto_print",
           "normalize_engineer_notation", "normalize_engineer_notation_safe",
           "normalize_numeric_verify_unit", "SplitResult", "normalize_timespan",
           "normalize_numeric_args", "returns_unit"]

@dataclass
class UnitSplitResult:
    remainder: str = ''
    unit_prefix: str = ''
    unit: str = ''

@dataclass
class SplitResult:
    prefix: str = ''
    number: str = ''
    unit_prefix_char: str = ''
    unit_prefix: str = ''
    unit: str = ''

@dataclass
class NormalizeResult:
    prefix: str = ''
    value: float = 0.0
    original_number: float = 0.0
    unit_prefix: str = ''
    unit: str = ''
    # Multiplier from prefix
    prefix_multiplier: float = 1.0

def _default_unit_prefix_map(include_length_unit_prefixes=False):
    """
    The default unit prefix to exponent mapping
    """
    unit_prefixes = {
        'y': -24,
        'z': -21,
        'a': -18,
        'f': -15,
        'p': -12,
        'n': -9,
        'µ': -6, 'μ': -6, 'u': -6,  # micro variants
        'm': -3,
        # No unit prefix for base unit (exponent 0)
        'k': 3,
        'M': 6,
        'G': 9,
        'T': 12,
        'E': 15,
        'Z': 18,
        'Y': 21
    }
    
    if include_length_unit_prefixes:
        unit_prefixes.update({
            'c': -2,  # e.g. centimeter
            'd': -1   # e.g. decimeter
        })
    
    return unit_prefixes

def _length_units(include_m=False):
    """
    All known length units.
    "m" is not included by default due to ambiguity
    with the milli-"m" (mg = milligram)
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

def _area_units() -> Set[str]:
    """
    All known area units (compact symbols only).
    See also Area.py
    """
    units = set([
        # Area units
        'm²', 'm^2',
        'in²', 'in^2',
        'ft²', 'ft^2', 
        'yd²', 'yd^2',
        'acre', 'ha', 'are',
        'barn', 'b',
        # NOTE: Do not list SI-prefixed units such as cm² here!
        # These must be parsed as m² with SI prefix "c" etc.
    ])
    return units

def _area_unit_aliases() -> Dict[str, str]:
    """
    Maps verbose area unit names to their compact symbols.
    """
    return {
        # Square inch aliases
        'square inch': 'in²',
        'square inches': 'in²', 
        'sq in': 'in²',
        
        # Square foot aliases
        'square foot': 'ft²',
        'square feet': 'ft²', 
        'sq ft': 'ft²',
        
        # Square yard aliases  
        'square yard': 'yd²',
        'square yards': 'yd²',
        'sq yd': 'yd²',
        
        # Other area aliases
        'acres': 'acre',
        'hectare': 'ha', 
        'hectares': 'ha', 
        'hectars': 'ha', 
        'ares': 'are',
        'barns': 'barn',
        'square meter': 'm²',
        'square meters': 'm²', 
        'sq m': 'm²', 
        'sqm': 'm²',
        'm^2': 'm²',  # Caret notation alias
        
        # SI prefixed meter aliases
        'square millimeter': 'mm²',
        'square millimeters': 'mm²', 
        'square mm': 'mm²', 
        'sq mm': 'mm²', 
        'mm sq': 'mm²', 
        'mm squared': 'mm²', 
        'millimeter squared': 'mm²',
        'mm^2': 'mm²',  # Caret notation alias
        
        'square centimeter': 'cm²',
        'square centimeters': 'cm²', 
        'square cm': 'cm²', 
        'sq cm': 'cm²', 
        'cm sq': 'cm²', 
        'cm squared': 'cm²', 
        'centimeter squared': 'cm²', 
        'centimeters squared': 'cm²',
        'cm^2': 'cm²',  # Caret notation alias
        
        'square decimeter': 'dm²',
        'square decimeters': 'dm²', 
        'square dm': 'dm²', 
        'sq dm': 'dm²', 
        'dm sq': 'dm²', 
        'dm squared': 'dm²', 
        'decimeter squared': 'dm²', 
        'decimeters squared': 'dm²',
        'dm^2': 'dm²',  # Caret notation alias
        
        'square micrometer': 'µm²',
        'square micrometers': 'µm²', 
        'square µm': 'µm²',
        'sq µm': 'µm²', 
        'sq um': 'µm²', 
        'um sq': 'µm²', 
        'µm sq': 'µm²', 
        'µm squared': 'µm²', 
        'micrometer squared': 'µm²', 
        'micrometers squared': 'µm²',
        'µm^2': 'µm²',  # Caret notation alias
        'um^2': 'µm²',  # Caret notation alias
        
        'square nanometer': 'nm²',
        'square nanometers': 'nm²', 
        'square nm': 'nm²', 
        'sq nm': 'nm²', 
        'nm sq': 'nm²', 
        'nm squared': 'nm²', 
        'nanometers squared': 'nm²',
        'nm^2': 'nm²',  # Caret notation alias
        
        'square kilometer': 'km²',
        'square kilometers': 'km²', 
        'square km': 'km²', 
        'sq km': 'km²', 
        'km squared': 'km²', 
        'km sq': 'km²', 
        'kilometers sq': 'km²', 
        'kilometers squared': 'km²',
        'km^2': 'km²',  # Caret notation alias
    }


def _default_units(include_m=False) -> Set[str]:
    return {
        # NOTE: These Ω symbols are NOT identical !
        'F', 'A', 'Ω', 'Ω', 'W', 'H', 'C', 'K', 'Hz', 'V', 'J', 'S',
        'R', # Ohms, but without having to copynpaste the Ω symbol
        # Time
        's', 'h', 'min',
        # Fraction
        'ppm', 'ppb', '%',
        # Lighting
        'lm', 'lx', 'cd',
        # Composite units
        'C/W', '€/km', '€/m',
        # Currencies
        '€', '$', '元', '﷼', '₽', '௹', '૱', '₺', 'Zł', '₩', '¥'
    }


def _default_timespan_units():
    return {
        # Attoseconds
        'as': 1e-18,
        'asec': 1e-18,
        'asecs': 1e-18,
        'attosecond': 1e-18,
        'attoseconds': 1e-18,
        # Femtoseconds
        'fs': 1e-15,
        'fsec': 1e-15,
        'fsecs': 1e-15,
        'femtosecond': 1e-15,
        'femtoseconds': 1e-15,
        # Picoseconds
        'ps': 1e-12,
        'psec': 1e-12,
        'psecs': 1e-12,
        'picosecond': 1e-12,
        'picoseconds': 1e-12,
        # Nanoseconds
        'ns': 1e-9,
        'nsec': 1e-9,
        'nsecs': 1e-9,
        'nanosecond': 1e-9,
        'nanoseconds': 1e-9,
        # Microseconds
        'µs': 1e-6,
        'us': 1e-6,
        'µsec': 1e-6,
        'usec': 1e-6,
        'microsecond': 1e-6,
        'microseconds': 1e-6,
        'µsecond': 1e-6,
        # Milliseconds
        'ms': 0.001,
        'millisecond': 0.001,
        'milliseconds': 0.001,
        # seconds
        's': 1,
        'sec': 1,
        'secs': 1,
        'second': 1,
        'seconds': 1,
        # Minutes
        'm': 60,
        'min': 60,
        # hours
        'h': 3600,
        'hour': 3600,
        'hours': 3600,
        # days
        'd': 86400,
        'day': 86400,
        'days': 86400,
        # weeks
        'w': 604800,
        'week': 604800,
        'weeks': 604800,
        # Months (we're using the average duration of a month)
        'mo': 2629746,  # 1/12th of a year, see below for the definition of a year
        'month': 2629746,
        'months': 2629746,
        # years (365.2425 days on average)
        'y': 31556952,
        'year': 31556952,
        'years': 31556952,
        # Decades
        'decade': 315569520,
        'decades': 315569520,
        # Centuries
        'century': 3155695200,
        'centuries': 3155695200,
        # Millenia
        'millenium': 31556952000,
        'millenia': 31556952000,
        # Megayears
        'My': 31556952000,
        'Myr': 31556952000,
        'Myrs': 31556952000,
        # Gigayears
        'Gy': 31556952000000,
        'Gyr': 31556952000000,
        'Gyrs': 31556952000000,
        # Terayears
        'Ty': 31556952000000000,
        'Tyr': 31556952000000000,
        'Tyrs': 31556952000000000,
    }

def _default_prefixes():
    return ["Δ", "±"]

def _default_unit_prefixes():
    return ["°"]

# Valid unit designators. Ensure no SI suffix is added here
_numeric_allowed = set("+0123456789-e.")

def none_to_nan(value):
    """
    Convert None to NaN, otherwise return the value unchanged.
    This is useful for normalizing values in arrays.
    """
    # NOTE: string is iterable, so we need to check for that first
    if isinstance(value, str):
        return np.nan if value.strip() == '' else value.strip()
    # NOTE: NormalizeResult is a namedtuple i.e. iterable, so we need to handle it separately
    if isinstance(value, NormalizeResult):
        return NormalizeResult(
            prefix=value.prefix,
            value=none_to_nan(value.value),
            unit_prefix=value.unit_prefix,
            unit=value.unit,
            
        )
    if isinstance(value, Iterable):
        # If it's an iterable, convert each element
        return [none_to_nan(elem) for elem in value]
    # Else: Assume a simple value
    if value is None:
        return np.nan
    return value

class EngineerIO(object):
    instance: Optional["EngineerIO"] = None
    length_instance: Optional["EngineerIO"] = None
    area_instance: Optional["EngineerIO"] = None
    """
    Default instance, used for global functions. Initialized on first use

    Note: ppm, ppb and % are special 'units' that are handled separately.
    """
    def __init__(self, units=_default_units(),
                 prefixes=_default_prefixes(),
                 unit_prefixes=_default_unit_prefixes(),
                 unit_prefix_map=_default_unit_prefix_map(),
                 timespan_units=_default_timespan_units(),
                 unit_aliases=None):
        """
        Initialize a new EngineerIO instance with default or custom unit prefixes

        Parameters:
        -----------
        units : iterable of strings
            An iterable of valid units (1-char or 2-char)
        prefixes : string
            A list of prefixes that are ignored (available via normalize()).
            Constraint: prefixes ∩ units == ∅
        unit_prefixes : string
            A list of prefixes that ignored (available via normalize()).
            Constraint: unitPrefixes ∩ units == ∅
        unit_prefix_map : dict
            Maps unit prefix strings to their decimal exponents.
            For generating strings from numbers, the first unit prefix in each nested list is  preferred
        unit_aliases : dict
            Maps unit alias strings to their canonical unit names
        """
        self.units = set(units)
        self.unit_prefix_map = unit_prefix_map
        self.timespan_units = timespan_units
        self.unit_aliases = unit_aliases or {}
        # Build prefix regex
        _prefix_set = "|".join(re.escape(pfx) for pfx in prefixes)
        self.prefix_re = re.compile('^(' + _prefix_set + ')+')
        # Build unit prefix regex
        __unitprefix_set = "|".join(re.escape(pfx) for pfx in unit_prefixes)
        self.unit_prefix_re = re.compile(f'({__unitprefix_set})+$') # $: Matched at end of numeric part
        # Unit prefixes will only be used in strip, so we can strip spaces in one go.
        self.strippable = " \t\n"
        # Compute maps
        self.all_unit_prefixes = set(self.unit_prefix_map.keys())
        self._recompute_unit_prefix_maps()
        # Compile unit alias regex
        self._compile_unit_alias_regex()
        # Compile units regex
        self._compile_units_regex()
        # Compile unit prefix regex
        self._compile_unit_prefix_suffix_regex()

    def _recompute_unit_prefix_maps(self):
        """
        Recompute the exponent -> unit prefix map from the unit prefix -> exponent map
        """
        # Direct mapping from unit prefix to exponent already exists in self.unit_prefix_map
        # Create the inverse mapping from exponent to unit prefix
        self.exp_unit_prefix_map = {}  # Key: exp // 3, Value: unit prefix
        self.unit_prefix_exp_map = {'': 0}  # Key: unit prefix, value: exponent (empty string for no unit prefix)
        
        # Copy the unit prefix map and add it to unit_prefix_exp_map
        for unit_prefix, exponent in self.unit_prefix_map.items():
            self.unit_prefix_exp_map[unit_prefix] = exponent
            # For the exp_unit_prefix_map, use the exponent divided by 3 as key
            exp_key = exponent // 3
            # Only store the first unit prefix for each exponent (for formatting)
            if exp_key not in self.exp_unit_prefix_map:
                self.exp_unit_prefix_map[exp_key] = unit_prefix
        
        # Add empty unit prefix for base unit (exponent 0)
        self.exp_unit_prefix_map[0] = ""
        
        # Compute min/max SI value
        if self.exp_unit_prefix_map:
            self.exp_map_min = min(self.exp_unit_prefix_map.keys())
            self.exp_map_max = max(self.exp_unit_prefix_map.keys())
        else:
            self.exp_map_min = 0
            self.exp_map_max = 0
            
    def _generate_unit_alias_pattern(self):
        """
        Generate a regex pattern to match unit aliases at the end of strings.
        Returns pattern string in format: "(alias1|alias2|...)$"
        """
        if not self.unit_aliases:
            return None
        
        # Sort aliases by length (longest first) to ensure proper matching
        # e.g. "square millimeters" should match before "millimeters"
        sorted_aliases = sorted(self.unit_aliases.keys(), key=len, reverse=True)
        
        # Escape each alias for regex and join with |
        escaped_aliases = [re.escape(alias) for alias in sorted_aliases]
        return f"({'|'.join(escaped_aliases)})$"

    def _compile_unit_alias_regex(self):
        """
        Compile a regex pattern to match unit aliases at the end of strings.
        """
        pattern = self._generate_unit_alias_pattern()
        if pattern is None:
            self.unit_alias_regex = None
            return
        
        # NOTE: Needs to be case-sensitive for some special units
        self.unit_alias_regex = re.compile(pattern, flags=re.UNICODE)

    def _generate_units_pattern(self):
        """
        Generate a regex pattern to match units at the end of strings.
        Returns pattern string in format: "(unit1|unit2|...)$"
        """
        if not self.units:
            return None
        
        # Sort units by length (longest first) to ensure proper matching
        # e.g. "Angstrom" should match before "A"
        sorted_units = sorted(self.units, key=len, reverse=True)
        
        # Escape each unit for regex and join with |
        escaped_units = [re.escape(unit) for unit in sorted_units]
        return f"({'|'.join(escaped_units)})$"

    def _compile_units_regex(self):
        """
        Compile a regex pattern to match units at the end of strings.
        """
        pattern = self._generate_units_pattern()
        if pattern is None:
            self.units_regex = None
            return
        
        # NOTE: Needs to be case-sensitive for some special units
        self.units_regex = re.compile(pattern, flags=re.UNICODE)

    def _compile_unit_prefix_suffix_regex(self):
        """
        Compile a regex pattern to match unit prefixes at the end of strings.
        """
        if not self.all_unit_prefixes:
            self.unit_prefix_suffix_regex = None
            return
        
        # Sort unit prefixes by length (longest first) to ensure proper matching
        sorted_prefixes = sorted(self.all_unit_prefixes, key=len, reverse=True)
        
        # Escape each prefix for regex and join with |
        escaped_prefixes = [re.escape(prefix) for prefix in sorted_prefixes]
        pattern = f"({'|'.join(escaped_prefixes)})$"
        
        # NOTE: Needs to be case-sensitive for unit prefixes
        self.unit_prefix_suffix_regex = re.compile(pattern, flags=re.UNICODE)

    def _resolve_unit_alias(self, unit):
        """
        Resolve a unit alias to its canonical form.
        """
        return self.unit_aliases.get(unit, unit)

    def has_any_unit_prefix(self, s):
        """
        Check if any suffix of the string is a unit prefix.
        Returns a tuple (has_prefix, unit_prefix_char, remainder) where:
        - has_prefix: True if a unit prefix suffix was found
        - unit_prefix_char: the unit prefix character found (or empty string)
        - remainder: the string with the unit prefix removed (or original string)
        """
        if not self.unit_prefix_suffix_regex:
            return False, "", s
        
        match = self.unit_prefix_suffix_regex.search(s)
        if match:
            unit_prefix_char = match.group(1)
            remainder = s[:match.start()]
            return True, unit_prefix_char, remainder
        
        return False, "", s

    def split_input(self, s):
        """
        Separate a string into a number, suffix and unit plus prefixes.
        Does not try to parse the numbers.
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
        orig_str = s
        # Remove thousands separator & ensure dot is used
        s = normalize_interpunctation(s)
        # Split off unit: "120kV" => "120k", "V"
        split_result = self.split_unit(s)
        print("Split result", split_result)
        # Print remainder
        s = split_result.remainder
        s = s.replace(" ", "")
        # Try to split prefix
        prefix_hit = self.prefix_re.search(s)
        if prefix_hit:
            # Get actual prefix (returned later)
            prefix = prefix_hit.group(0)
            # Remove prefix
            s = self.prefix_re.sub("", s)
        else:
            prefix = ""
        # Check string
        if not s:
            raise ValueError("Can't split empty string")
        # Try to find SI unit prefix using suffix checking
        string_is_suffixed_by_unit_prefix, unit_prefix_char, remainder = self.has_any_unit_prefix(s)
        if string_is_suffixed_by_unit_prefix: # e.g. "2.5k" is terminated by "k"
            s = remainder
        else:  # Try to find unit prefix anywhere, e.g. in the middle
            # Check every character in the string for being a SI prefix
            is_unit_prefix_list: List[bool] = [(ch in self.all_unit_prefixes) for ch in s]
            # Check if first character in the string is a unit prefix ("k12.5")
            if is_unit_prefix_list[0]:
                # "k12" is not a valid engineer string
                raise FirstCharacterInStringIsUnitPrefixException(f"The first character in '{s}', that is '{s[0]}' is registered as a SI prefix, hence the meaning of that string is not clear")
            # Handle various cased depending on number of prefix characters in the string
            unit_prefix_count = sum(is_unit_prefix_list)
            if unit_prefix_count == 1: # Exactly one SI prefix character 8but it's not at the end
                unit_prefix_index = is_unit_prefix_list.index(True)
                unit_prefix_char = s[unit_prefix_index]
            elif unit_prefix_count > 1:
                # This case occurs e.g. if you use pnJ, in which case it's not clear what that means.
                # (pico-nano-Joules??!?)
                # Special rule for "m" (meters): "cm" must be a valid prefix, plus unit
                # So if exactly 2 valid prefixes are detected, and the the first one is 
                detected_prefixes = [ch for ch in s if ch in self.all_unit_prefixes]
                if len(detected_prefixes) == 2 and detected_prefixes[-1] == 'm':
                    # for "cm", use "c"
                    unit_prefix_index = is_unit_prefix_list.index(True)
                    unit_prefix_char = detected_prefixes[0]
                    # Leave unitPrefixIndex as the first True index
                else: # Special rule does not apply => fail!
                    raise MultipleUnitPrefixesException(f"More than one SI unit prefix in the string '{s}'. Orig str: {orig_str}, Detected unit prefixes: '{detected_prefixes}'")
            else: # unit_prefix_count == 0
                unit_prefix_char = ""
                unit_prefix_index = -1
            # Perform additional checks & conversions for prefix-as-decimal-separator
            if unit_prefix_char:
                # Unit prefix found in the middle of the string such as "1k25"
                # Check if unit prefix is between two digits (prefix-as-decimal-separator)
                unit_prefix_char = s[unit_prefix_index]
                is_between_digits = (unit_prefix_index > 0 and unit_prefix_index < len(s) - 1 and
                                    s[unit_prefix_index - 1].isdigit() and s[unit_prefix_index + 1].isdigit())
                
                if is_between_digits:
                    # Unit prefix-as-decimal-separator --> there must be no other decimal separator
                    if "." in s:  # Comma-to-dot conversion already handled by normalize_interpunctation
                        raise ValueError(f"Unit prefix as decimal separator, but dot is also in string: {s}")
                    # "1k25" => "1.25", also save "k" as unit prefix char
                    unit_prefix_char = s[unit_prefix_index]
                    s = s.replace(unit_prefix_char, ".")

        s = s.strip(self.strippable)
        # Final check: After applying all rules, the string should be all numbers
        if not all((ch in _numeric_allowed for ch in s)):
            raise RemainderOfStringContainsNonNumericCharacters(f"'{s}'. Orig str: {orig_str}, Detected unit_prefix '{unit_prefix_char}', split result {split_result}")
        return SplitResult(
            prefix=prefix,
            number=s,
            unit_prefix_char=unit_prefix_char,
            unit_prefix=split_result.unit_prefix,
            unit=split_result.unit
        )

    def split_unit(self, s):
        """
        Split a string into (remainder, unit).
        Only units in the units set are recognized
        unit may be '' if no unit is recognized
        """
        # Fallback for strings which are too short
        if len(s) <= 1:
            return UnitSplitResult(s, '', '')
        # Check for unit aliases first
        if self.unit_alias_regex:
            alias_match = self.unit_alias_regex.search(s)
            if alias_match:
                alias = alias_match.group(1)
                canonical_unit = self._resolve_unit_alias(alias)
                # NOTE: We need to replace the unit alias by the unit explicitly
                # (and let the rest of the code handle it).
                # This is since the aliased unit may contain a SI prefix such as
                # "sq cm" => "cm²"
                # Hence, we need to replace the matched alias by the unit 
                # in the string, and the safest way to do that is to use the match indexes
                start_idx = alias_match.start(1)
                end_idx = alias_match.end(1)
                # Modify the string in-place
                s = s[:start_idx] + canonical_unit + s[end_idx:]
                # Now continue with the loop
        
        # Check for units using compiled regex
        if self.units_regex:
            unit_match = self.units_regex.search(s)
            if unit_match:
                unit = unit_match.group(1)
                remainder = s[:unit_match.start()].strip()
                print("Found unit", unit)
                # Remove unit prefix, if any
                unit_prefix_hit = self.unit_prefix_re.search(remainder)
                if unit_prefix_hit:
                    # Get actual prefix (returned later)
                    unit_prefix = unit_prefix_hit.group(0)
                    # Remove unit_prefix
                    remainder = self.unit_prefix_re.sub("", remainder)
                else:
                    unit_prefix = ""
                # Remove extra whitespace
                remainder = remainder.rstrip(self.strippable)
                return UnitSplitResult(remainder, unit_prefix, unit)
        
        # Fallback: No unit found
        value_str, unit = s, ''
        # Remove extra whitespace
        value_str = value_str.rstrip(self.strippable)
        # Remove unit prefix, if any
        unit_prefix_hit = self.unit_prefix_re.search(value_str)
        if unit_prefix_hit:
            # Get actual prefix (returned later)
            unit_prefix = unit_prefix_hit.group(0)
            # Remove unit_prefix
            value_str = self.unit_prefix_re.sub("", value_str)
        else:
            unit_prefix = ""
        # Remove extra whitespace
        value_str = value_str.rstrip(self.strippable)
        return UnitSplitResult(value_str, unit_prefix, unit)

    def normalize(self, s, encoding="utf8", prefix_exponent=1.0):
        """
        Converts an engineer's input of a wide variety of formats to a numeric
        value.

        Returns a NormalizeResult() or None if the conversion could not be performed.
        
        prefix_exponent is used for converting area & volume units etc

        See split_input() for further details on supported formats
        """
        # Scalars get returned directly
        if isinstance(s, (int, float)):
            return NormalizeResult('', s, '', '', '', 1.0)
        # Make sure it's a decoded string
        if isinstance(s, bytes):
            s = s.decode(encoding)
        # Handle lists / array
        if isinstance(s, (list, tuple, np.ndarray)):
            return [self.normalize(elem) for elem in s]
        # Perform splitting
        split_result = self.split_input(s.strip())
        mul = (10 ** self.unit_prefix_exp_map[split_result.unit_prefix_char])**prefix_exponent if split_result.unit_prefix_char else 1
        # Handle ppm and ppb: They are listed as units
        unit = split_result.unit
        if unit == '%':
            mul /= 100
            unit = ''
        elif unit == 'ppm':
            mul /= 1e6
            unit = ''
        elif unit == 'ppb':
            mul /= 1e9
            unit = ''
        num = float(split_result.number)
        return NormalizeResult(
            prefix=split_result.prefix,
            value=num * mul,
            prefix_multiplier=mul,
            original_number=num,
            unit_prefix=split_result.unit_prefix,
            unit=split_result.unit
        )

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
        Format v using SI unit_prefixes with optional units.
        Produces a string with 3 visible digits.
        """
        if unit is None:
            unit = ""
        # Handle NaN
        if np.isnan(v):
            return _format_with_suffix(
                v,
                unit,
                significant_digits=significant_digits
            )
        #Unit_prefix map is indexed by one third of the decadic logarithm.
        exp = 0 if v == 0. else math.log(abs(v), 10.)
        unit_prefixMapIdx = int(math.floor(exp / 3.))
        #Ensure we're in range
        if not self.exp_map_min < unit_prefixMapIdx < self.exp_map_max:
            raise ValueError(f"Value out of range: {v}")
        #Pre-multiply the value
        v = v * (10.0 ** -(unit_prefixMapIdx * 3))
        #Delegate the rest of the task to the helper
        return _format_with_suffix(
            v,
            self.exp_unit_prefix_map[unit_prefixMapIdx] + unit,
            significant_digits=significant_digits
        )

    def print(self, v, unit="", significant_digits=3):
        """
        Like format_value, but also prints the value
        """
        s = self.format(v, unit, significant_digits)
        print(s) # This is not a debug print.
        return s


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
        return multiplier, self.exp_unit_prefix_map[suffix_idx]
    
    def extract_return_unit(self, fn):
        """
        Extract the return unit from a function's annotation.
        """
        unit = getattr(fn, "_returns_unit", None)
        # Special rule for functools.partial or similar
        if unit is None and hasattr(fn, "func"):
            # Access next function level inside possibly nested partials
            # NOTE: Any of the nested function levels may have the annotation!
            return self.extract_return_unit(fn.func)
        return unit

    def auto_format(self, fn, *args, significant_digits=3, **kwargs):
        """
        Auto-format a value by leveraging a custom @returns_unit annotation.
        The function's return value is expected to be annotated with @returns_unit("unit").
        """
        # Try to get the direct function's return value unit
        unit = self.extract_return_unit(fn)
        if unit is None:
            raise UnannotatedReturnValueError("Function must be annotated with @returns_unit('...')")
        return self.format(fn(*args, **kwargs), unit=unit, significant_digits=significant_digits)

    def auto_print(self, *args, **kwargs):
        print(self.auto_format(*args, **kwargs))
        
    def normalize_iterable(self, arg, func):
        """
        Normalize an iterable (works for lists, tuples, numpy arrays and generators)
        """
        size = len(arg)
        resize_step = 1000 # Default size if size is indeterminate
        size_indeterminate = size < 1
        # If size is invalid, we use a default array of size 100 to prevent frequent reallocation
        ret = np.zeros(resize_step if size_indeterminate else size, dtype=float)
        n = 0
        for i, elem in enumerate(arg):
            if i > len(arg):
                # Resize arg to at least [i]
                # This might be slow, but better than crashing, and
                # we can't expect the generator to be iterable twice
                # NOTE: This resizes only every 1000th element
                ret = np.resize(ret, i + resize_step)
            func_result = func(elem)
            # If it has .value, return .value, otherwise return the value directly
            if hasattr(func_result, 'value'):
                ret[i] = func_result.value
            else:
                ret[i] = func_result
            n = i + 1
        if size_indeterminate:
            # If the size was indeterminate, we return a view of the array
            return ret[:n]
        return ret

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
            if v is None:
                return None
            if isinstance(v, NormalizeResult):
                return v.value
            else:
                return v
        # It's an iterable
        print(arg, self.safe_normalize(arg), none_to_nan(self.safe_normalize(arg)))
        return self.normalize_iterable(arg, func=lambda v: none_to_nan(self.safe_normalize(v)))

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
            return self.normalize(arg).value
        # It's an iterable
        return self.normalize_iterable(arg, func=self.normalize)

    def normalize_numeric_verify_unit(self, arg, reference):
        """
        Normalize a value. If it is a string
        verify if its unit matches the reference unit.
        """
        if arg is None:
            raise ValueError("Can't normalize None")

        # Scalars get returned directly
        if isinstance(arg, (int, float, np.generic)):
            return arg

        # If it's stringlike, apply directly
        if isinstance(arg, (str, bytes)):
            normalize_result = self.normalize(arg)
            # Check if unit matches (it's also considered a match if there is no unit at all)
            if normalize_result.unit and normalize_result.unit != reference.unit:
                raise InvalidUnitInContextException(f"Invalid unit: Expected {reference} but found {normalize_result.unit} in source string '{arg}'")
            return normalize_result.value
        # It's an iterable
        return self.normalize_iterable(arg, func=self.normalize)
    
    def normalize_timespan(self, arg: str | bytes | int | float | np.generic | np.ndarray) -> int | float | np.generic | np.ndarray:
        """
        Normalize a given timespan to SI units (seconds).
        Numeric inputs are assumed to be in seconds.
        """
        if isinstance(arg, bytes):
            arg = arg.decode("utf8")
        if isinstance(arg, (int, float, np.generic)):
            return arg # Already a number. Just return!
        elif isinstance(arg, (str)):
            s, unit = partition_at_numeric_to_nonnumeric_boundary(arg) # Remove unit
            s, unit = s.strip(), unit.strip()
            if not s:
                raise ValueError(f"Empty value in timespan: {arg}")
            if not unit: # Assume seconds (SI unit of time)
                return float(s)
            # Check if unit exists in timespan_units
            if unit not in self.timespan_units:
                raise ValueError(f"Invalid timespan unit '{unit}' in '{arg}'. Expected one of {list(self.timespan_units.keys())}")
            return float(s) * self.timespan_units[unit]
        elif isinstance(arg, (np.ndarray, list)):
            return np.vectorize(self.normalize_timespan)(arg)
        else:
            raise ValueError(f"Unsupported type for normalization: {type(arg)}")

# Initialize global instance
EngineerIO.instance = EngineerIO()
EngineerIO.length_instance = EngineerIO(
    units=_default_units(include_m=True),
    unit_prefix_map=_default_unit_prefix_map(include_length_unit_prefixes=True)
)
EngineerIO.area_instance = EngineerIO(
    units=_area_units(),
    unit_aliases=_area_unit_aliases(),
    unit_prefix_map=_default_unit_prefix_map(include_length_unit_prefixes=True)
)

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
    abs_v = abs(v)
    if np.isnan(v):
        res = "-"
    elif abs_v < 1.0:
        res = f"{v:.{significant_digits - 0}f}"
    elif abs_v < 10.0:
        res = f"{v:.{significant_digits - 1}f}"
    elif abs_v < 100.0:
        res = f"{v:.{significant_digits - 2}f}"
    else:  # Should only happen if v < 1000
        res = str(int(round(v)))
    #Avoid appending whitespace if there is no suffix
    return f"{res} {suffix}" if suffix else res

def normalize_engineer_notation(s, encoding="utf8"):
    return EngineerIO.instance.normalize(s, encoding=encoding)

def format_value(v, unit="", significant_digits=3):
    return EngineerIO.instance.format(v, unit, significant_digits=significant_digits)

def print_value(v, unit="", significant_digits=3):
    return EngineerIO.instance.print(v, unit, significant_digits=significant_digits)

def normalize_numeric_verify_unit(self, arg, reference):
    return EngineerIO.instance.normalize_numeric_verify_unit(arg, reference)

def normalize_engineer_notation_safe(v, unit=""):
    return EngineerIO.instance.safe_normalize(v, unit)

def normalize_numeric(v):
    return EngineerIO.instance.normalize_numeric(v)

def normalize(v):
    return EngineerIO.instance.normalize(v)

def normalize_timespan(v: str | bytes | int | float | np.generic | np.ndarray) -> int | float | np.generic | np.ndarray:
    return EngineerIO.instance.normalize_timespan(v)

def auto_format(v, *args, **kwargs):
    return EngineerIO.instance.auto_format(v, *args, **kwargs)

def auto_print(*args, **kwargs):
    return EngineerIO.instance.auto_print(*args, **kwargs)

def returns_unit(unit):
    """
    Decorator to annotate a function with a custom return unit string.
    Usage: @returns_unit("A")
    """
    def decorator(fn):
        fn._returns_unit = unit
        return fn
    return decorator

def normalize_numeric_args(func=None, *, exclude=None):
    """
    Decorator that applies normalize_numeric to all arguments (args & kwargs) 
    of the decorated function before calling it.
    
    This allows functions to accept engineer notation strings and automatically
    convert them to numeric values.
    
    Parameters:
    -----------
    exclude : list of str, optional
        List of parameter names that should not be normalized
    
    Example:
        @normalize_numeric_args
        def add(a, b):
            return a + b
        
        result = add("1.5k", "2.3k")  # Will convert to add(1500.0, 2300.0)
        
        @normalize_numeric_args(exclude=['unit'])
        def calculate(value, unit):
            return value  # value is normalized, unit is left as string
    """
    if exclude is None:
        exclude = []
    exclude_set = set(exclude)
    
    def decorator(func):
        # Get the function signature
        sig = inspect.signature(func)
        
        # Create new parameters with normalized default values
        new_params = []
        for param in sig.parameters.values():
            if param.name not in exclude_set and param.default != inspect.Parameter.empty and isinstance(param.default, str):
                # Normalize string default values
                try:
                    normalized_default = normalize_numeric(param.default)
                    new_param = param.replace(default=normalized_default)
                except:
                    # If normalization fails, keep the original default
                    new_param = param
            else:
                new_param = param
            new_params.append(new_param)
        
        # Create new signature with normalized defaults
        new_sig = sig.replace(parameters=new_params)

        def wrapper(*args, **kwargs):
            # Get parameter names from signature
            param_names = list(sig.parameters.keys())
            
            # Normalize positional arguments (skip excluded ones)
            normalized_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else None
                if param_name in exclude_set:
                    normalized_args.append(arg)
                else:
                    normalized_args.append(normalize_numeric(arg))
            normalized_args = tuple(normalized_args)
            
            # Normalize keyword arguments (skip excluded ones)
            normalized_kwargs = {}
            for key, value in kwargs.items():
                if key in exclude_set:
                    normalized_kwargs[key] = value
                else:
                    normalized_kwargs[key] = normalize_numeric(value)
            
            # Bind arguments to new signature to get all parameters with defaults applied
            bound_args = new_sig.bind(*normalized_args, **normalized_kwargs)
            bound_args.apply_defaults()
            
            # Call the original function with all normalized arguments (including defaults)
            return func(*bound_args.args, **bound_args.kwargs)
        
        # Preserve function metadata
        functools.update_wrapper(wrapper, func)
        wrapper.__signature__ = new_sig
        wrapper._returns_unit = getattr(func, "_returns_unit", None)
        
        return wrapper
    
    # Handle both @normalize_numeric_args and @normalize_numeric_args(exclude=[...])
    if func is None:
        return decorator
    else:
        return decorator(func)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default values and configurations for the EngineerIO library.
"""
from toolz import functoolz
from typing import Callable, Dict, List, Tuple, Union
from UliEngineering.EngineerIO.UnitInfo import UnitAlias, UnitInfo

__all__ = [
    'default_unit_prefixes',
    'default_si_prefix_map',
    'default_unit_infos',
    'default_interpunctation_transform_map',
]

def default_unit_prefixes() -> List[str]:
    return ["Δ", "±", "°"]

def default_si_prefix_map(include_length_unit_prefixes=False) -> Dict[str, float]:
    """
    Generate a dictionary mapping SI prefixes to their corresponding exponents (in powers of 10).
    Parameters
    ----------
    include_length_unit_prefixes : bool, optional
        If True, includes additional prefixes commonly used for length units:
        centi- ('c', 10^-2) and deci- ('d', 10^-1). Default is False.
    Returns
    -------
    dict
        A dictionary where keys are string prefixes and values are the corresponding
        exponents (powers of 10). For example, 'k' maps to 3 (representing kilo-, or 10^3).
    Notes
    -----
    - For the micro prefix, three different symbols are supported: 'µ', 'μ', and 'u'
    - The range of standard prefixes spans from yocto ('y', 10^-24) to yotta ('Y', 10^21)
    - The base unit (exponent 0) is not included in the dictionary
    """
    unit_prefixes = {
        'y': -24.0,
        'z': -21.0,
        'a': -18.0,
        'f': -15.0,
        'p': -12.0,
        'n': -9.0,
        'µ': -6.0, 'μ': -6.0, 'u': -6.0,  # micro variants
        'm': -3.0,
        # No unit prefix for base unit (exponent 0)
        'k': 3.0,
        'M': 6.0,
        'G': 9.0,
        'T': 12.0,
        'P': 15.0,
        'E': 18.0,
        'Z': 21.0,
        'Y': 24.0
    }
    
    if include_length_unit_prefixes:
        unit_prefixes.update({
            'c': -2.0,  # e.g. centimeter
            'd': -1.0   # e.g. decimeter
        })
    
    return unit_prefixes

def default_unit_infos() -> List[Union[UnitInfo, UnitAlias]]:
    """
    Returns the default list of UnitInfo and UnitAlias objects for standard engineering units.
    """
    return [
        # Electrical units
        UnitInfo('F', aliases=['Farad', 'farads']),  # Capacitance
        UnitInfo('A', aliases=['Amp', 'Amps', 'Ampere', 'amperes']),  # Current
        UnitInfo('Ω', aliases=['Ohm', 'Ohms', 'ohm', 'ohms', 'R', 'Ω']),  # Resistance
        UnitInfo('W', aliases=['Watt', 'Watts', 'watt', 'watts']),  # Power
        UnitInfo('H', aliases=['Henry', 'Henries', 'henry', 'henries']),  # Inductance
        UnitInfo('C', aliases=['Coulomb', 'coulombs']),  # Charge
        UnitInfo('V', aliases=['Volt', 'Volts', 'volt', 'volts']),  # Voltage
        UnitInfo('J', aliases=['Joule', 'Joules', 'joule', 'joules']),  # Energy
        UnitInfo('S', aliases=['Siemens', 'siemens']),  # Conductance
        UnitInfo('Hz', aliases=['Hertz', 'hertz']),  # Frequency

        # Temperature
        UnitInfo('K', aliases=['Kelvin', 'kelvin']),
        
        # Time units
        UnitInfo('s', aliases=['second', 'seconds', 'sec']),
        UnitInfo('h', aliases=['hour', 'hours', 'hr']),
        UnitInfo('min', aliases=['minute', 'minutes']),
        
        # Fraction/percentage units
        UnitInfo('ppm', aliases=['parts per million']),
        UnitInfo('ppb', aliases=['parts per billion']),
        UnitInfo('%', aliases=['percent', 'percentage']),
        
        # Lighting units
        UnitInfo('lm', aliases=['lumen', 'lumens']),
        UnitInfo('lx', aliases=['lux']),
        UnitInfo('cd', aliases=['candela', 'candelas']),
        
        # Composite units
        UnitInfo('C/W'),
        UnitInfo('€/km'),
        UnitInfo('€/m'),
        UnitInfo('F/m'),
        
        # Currency units
        UnitInfo('€', aliases=['Euro', 'Euros', 'euro', 'euros']),
        UnitInfo('$', aliases=['Dollar', 'Dollars', 'dollar', 'dollars', 'USD']),
        UnitInfo('元', aliases=['Yuan', 'yuan', 'CNY']),
        UnitInfo('﷼', aliases=['Riyal', 'riyal', 'SAR']),
        UnitInfo('₽', aliases=['Ruble', 'ruble', 'RUB']),
        UnitInfo('௹', aliases=['Rupee', 'rupee', 'INR']),
        UnitInfo('૱'),
        UnitInfo('₺', aliases=['Lira', 'lira', 'TRY']),
        UnitInfo('Zł', aliases=['Zloty', 'zloty', 'PLN']),
        UnitInfo('₩', aliases=['Won', 'won', 'KRW']),
        UnitInfo('¥', aliases=['Yen', 'yen', 'JPY']),
    ]

def replace_comma_dot(s: str) -> str:
    return s.replace(",", ".")
    
def default_interpunctation_transform_map() -> Dict[Tuple[bool, bool, bool], Callable]:
    return {
        # Found nothing or only point -> no modification required
        (False, False, False): functoolz.identity,
        (False, False, True): functoolz.identity,
        (False, True, False): functoolz.identity,
        (False, True, True): functoolz.identity,
        # Only comma -> replace and exit
        (True, False, False): replace_comma_dot,
        (True, False, True): replace_comma_dot,
        # Below this line: Both comma and dot found
        # Comma first => comma used as thousands separators
        (True, True, True): lambda s: s.replace(",", ""),
        # Dot first => dot used as thousands separator
        (True, True, False): lambda s: s.replace(".", "").replace(",", ".")
    }
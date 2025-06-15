#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A python script to calculate E96 resistor values
... and do other useful things with resistors, e.g.
connect them in parallel and serial fashions.

Originally published at techoverflow.net
"""
import itertools
import numpy as np
from UliEngineering.EngineerIO import normalize_numeric
from collections import namedtuple

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "e192", "e96", "e48", "e24", "e12", "e6", "resistor_range",
    "standard_resistors", "parallel_resistors",
    "current_through_resistor", "standard_resistors_in_range",
    "power_dissipated_in_resistor_by_current",
    "power_dissipated_in_resistor_by_voltage",
    "voltage_across_resistor",
    "series_resistors", "nearest_resistor",
    "resistor_by_voltage_and_current",
    "next_higher_resistor", "next_lower_resistor",
    "resistor_current_by_power", "ResistorTolerance",
    "resistor_tolerance", "resistor_value_by_voltage_and_power"
]

#
# Standard resistor sequences
# Source: https://en.wikipedia.org/wiki/E_series_of_preferred_numbers
#
class ESeries:
    """Container for standard E-series values as NumPy arrays"""
    
    E6 = np.array([1.0, 1.5, 2.2, 3.3, 4.7, 6.8])
    
    E12 = np.array([1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2])
    
    E24 = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])
    
    E48 = np.array([1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54,
                    1.62, 1.69, 1.78, 1.87, 1.96, 2.05, 2.15, 2.26, 2.37, 2.49,
                    2.61, 2.74, 2.87, 3.01, 3.16, 3.32, 3.48, 3.65, 3.83, 4.02,
                    4.22, 4.42, 4.64, 4.87, 5.11, 5.36, 5.62, 5.90, 6.19, 6.49,
                    6.81, 7.15, 7.50, 7.87, 8.25, 8.66, 9.09, 9.53])
    
    E96 = np.array([1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24,
                    1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58,
                    1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00,
                    2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55,
                    2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24,
                    3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
                    4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23,
                    5.36, 5.49, 5.62, 5.76, 5.90, 5.97, 6.04, 6.12, 6.19, 6.26,
                    6.34, 6.42, 6.49, 6.57, 6.65, 6.73, 6.81, 6.90, 6.98, 7.06,
                    7.15, 7.23, 7.32, 7.41, 7.50, 7.59, 7.68, 7.77, 7.87, 7.96,
                    8.06, 8.16, 8.25, 8.35, 8.45, 8.56, 8.66, 8.76, 8.87, 8.98,
                    9.09, 9.20, 9.31, 9.42, 9.53, 9.65, 9.76, 9.88])
    
    E192 = np.array([1.00, 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.09, 1.10, 1.11,
                     1.13, 1.14, 1.15, 1.17, 1.18, 1.20, 1.21, 1.23, 1.24, 1.26,
                     1.27, 1.29, 1.30, 1.32, 1.33, 1.35, 1.37, 1.38, 1.40, 1.42,
                     1.43, 1.45, 1.47, 1.49, 1.50, 1.52, 1.54, 1.56, 1.58, 1.60,
                     1.62, 1.64, 1.65, 1.67, 1.69, 1.72, 1.74, 1.76, 1.78, 1.80,
                     1.82, 1.84, 1.87, 1.89, 1.91, 1.93, 1.96, 1.98, 2.00, 2.03,
                     2.05, 2.08, 2.10, 2.13, 2.15, 2.18, 2.21, 2.23, 2.26, 2.29,
                     2.32, 2.34, 2.37, 2.40, 2.43, 2.46, 2.49, 2.52, 2.55, 2.58,
                     2.61, 2.64, 2.67, 2.71, 2.74, 2.77, 2.80, 2.84, 2.87, 2.91,
                     2.94, 2.98, 3.01, 3.05, 3.09, 3.12, 3.16, 3.20, 3.24, 3.28,
                     3.32, 3.36, 3.40, 3.44, 3.48, 3.52, 3.57, 3.61, 3.65, 3.70,
                     3.74, 3.79, 3.83, 3.88, 3.92, 3.97, 4.02, 4.07, 4.12, 4.17,
                     4.22, 4.27, 4.32, 4.37, 4.42, 4.48, 4.53, 4.59, 4.64, 4.70,
                     4.75, 4.81, 4.87, 4.93, 4.99, 5.05, 5.11, 5.17, 5.23, 5.30,
                     5.36, 5.42, 5.49, 5.56, 5.62, 5.69, 5.76, 5.83, 5.90, 5.97,
                     6.04, 6.12, 6.19, 6.26, 6.34, 6.42, 6.49, 6.57, 6.65, 6.73,
                     6.81, 6.90, 6.98, 7.06, 7.15, 7.23, 7.32, 7.41, 7.50, 7.59,
                     7.68, 7.77, 7.87, 7.96, 8.06, 8.16, 8.25, 8.35, 8.45, 8.56,
                     8.66, 8.76, 8.87, 8.98, 9.09, 9.20, 9.31, 9.42, 9.53, 9.65,
                     9.76, 9.88])

# Global series exposed for backward compatibility
e6 = tuple(ESeries.E6)
e12 = tuple(ESeries.E12)
e24 = tuple(ESeries.E24)
e48 = tuple(ESeries.E48)
e96 = tuple(ESeries.E96)
e192 = tuple(ESeries.E192)

@normalize_numeric_args
@returns_unit("A")
def current_through_resistor(resistor, voltage):
    """
    Compute the current that flows through a resistor
    using ohms law.

    Parameters
    ----------
    resistor : float or Engineer string
        The resistor in Ohms
    voltage : float or Engineer string
        The voltage across the resistor
    """
    return voltage / resistor

@normalize_numeric_args
@returns_unit("V")
def voltage_across_resistor(resistor, current):
    """
    Compute the voltage that is dropped across
    a resistor using ohms law.

    Parameters
    ----------
    resistor : float or Engineer string
        The resistor in Ohms
    current : float or Engineer string
        The current flowing through the resistor
    """
    return resistor * current

@normalize_numeric_args
@returns_unit("W")
def power_dissipated_in_resistor_by_current(resistor, current):
    """
    Compute the power that is dissipated in
    a resistor using P=I²R given
    its resistance and the current flowing through it

    Parameters
    ----------
    resistor : float or Engineer string
        The resistor in Ohms
    current : float or Engineer string
        The current flowing through the resistor
    """
    return np.abs(resistor * current * current)

@normalize_numeric_args
@returns_unit("W")
def power_dissipated_in_resistor_by_voltage(resistor, voltage):
    """
    Compute the power that is dissipated in
    a resistor using P=VI given
    its resistance and the current flowing through it

    Parameters
    ----------
    resistor : float or Engineer string
        The resistor in Ohms
    current : float or Engineer string
        The current flowing through the resistor
    """
    voltage = normalize_numeric(voltage)
    current = current_through_resistor(resistor, voltage)
    return np.abs(current * voltage)

def resistor_range(multiplicator, sequence=e96):
    """
    Get a single range of resistors of a given sequence,
    e.g. for 1k to <10k use multiplicator = 1000.
    """
    # Multiply ndarrays directly for performance
    if isinstance(sequence, np.ndarray):
        return sequence * multiplicator
    # Otherwise, use list comprehension
    return [r * multiplicator for r in sequence]

def standard_resistors(minExp=-1, maxExp=9, sequence=e96):
    """
    Get a list of all standard resistor values from 100mOhm up to 976 MΩ in Ω"""
    exponents = itertools.islice(itertools.count(minExp, 1), 0, maxExp - minExp)
    multiplicators = [10 ** x for x in exponents]
    return itertools.chain(*(resistor_range(r, sequence=sequence) for r in multiplicators))

def standard_resistors_in_range(min_resistor="1Ω", max_resistor="10MΩ", sequence=e96):
    """
    Get all standard resistor values in Ω between min_resistor and max_resistor 
    
    :return: list of resistor values in Ω
    """
    min_resistor = normalize_numeric(min_resistor)
    max_resistor = normalize_numeric(max_resistor)
    return [
        resistor for resistor in standard_resistors(sequence=sequence)
        if min_resistor <= resistor <= max_resistor
    ]

@returns_unit("Ω")
def nearest_resistor(value, sequence=e96):
    """
    Find the standard reistor value with the minimal difference to the given value
    """
    value = normalize_numeric(value)
    return min(standard_resistors(sequence=sequence), key=lambda r: abs(value - r))

@returns_unit("Ω")
def next_higher_resistor(value, sequence=e96):
    """
    Find the next higher standard resistor value
    """
    value = normalize_numeric(value)
    return min((r for r in standard_resistors(sequence=sequence) if r > value), default=None)

@returns_unit("Ω")
def next_lower_resistor(value, sequence=e96):
    """
    Find the next lower standard resistor value
    """
    value = normalize_numeric(value)
    return max((r for r in standard_resistors(sequence=sequence) if r < value), default=None)

@normalize_numeric_args
@returns_unit("Ω")
def parallel_resistors(*args):
    """
    Compute the total resistance of n parallel resistors and return
    the value in Ohms.
    """
    resistors = np.asarray(list(args))
    # If there is no resistor, the value of the string is infinite Ohms
    if len(resistors) == 0:
        return np.inf
    # If there is any zero resistor, the value of the string is zero Ohms
    if len(np.nonzero(resistors)[0]) != len(resistors):
        return 0.0
    return 1.0 / np.sum(np.reciprocal(resistors.astype(float)))

@normalize_numeric_args
@returns_unit("Ω")
def series_resistors(*args):
    """
    Compute the total resistance of n parallel resistors and return
    the value in Ohms.
    """
    resistors = list(args)
    return sum(resistors)

@normalize_numeric_args
@returns_unit("Ω")
def resistor_by_voltage_and_current(voltage, current):
    """
    Compute the resistance value in Ohms that draws the given amount of
    current if the given voltage is across it.
    """
    return voltage / current

@normalize_numeric_args
@returns_unit("A")
def resistor_current_by_power(resistor, power):
    """
    Compute the current that flows through a resistor
    given its resistance and the power dissipated in it.
    """
    return np.sqrt(power / resistor)

ResistorTolerance = namedtuple("ResistorTolerance", ["lower", "nominal", "upper"])

@normalize_numeric_args
def resistor_tolerance(resistance, tolerance="1%") -> ResistorTolerance:
    """
    Compute the lower, nominal and upper bound of a resistor value
    given the nominal value and the tolerance.
    """
    return ResistorTolerance(
        lower=resistance - (resistance * tolerance),
        nominal=resistance,
        upper=resistance + (resistance * tolerance)
    )

@normalize_numeric_args
@returns_unit("Ω")
def resistor_value_by_voltage_and_power(voltage, power):
    """
    Compute resistor value given voltage across it and power dissipated.
    
    Uses the formula: R = V² / P
    
    This function does not perform any validity checks on the inputs,
    hence voltage and/or power can be zero or negative.
    
    Parameters
    ----------
    voltage : float
        Voltage across the resistor in volts
    power : float
        Power dissipated by the resistor in watts
        
    Returns
    -------
    float
        Resistance value in ohms
        
    Raises
    ------
    ValueError
        If power is zero or negative
    """
    return voltage ** 2 / power
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
from UliEngineering.Units import Unit

__all__ = ["e96", "e48", "e24", "e12", "resistor_range",
           "standard_resistors", "parallel_resistors",
           "current_through_resistor",
           "power_dissipated_in_resistor_by_current",
           "power_dissipated_in_resistor_by_voltage",
           "voltage_across_resistor",
           "series_resistors", "nearest_resistor",
           "resistor_by_voltage_and_current"]

# Standard resistor sequences
e96 = np.asarray([
       1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40,
       1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00,
       2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87,
       2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
       4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90,
       6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45,
       8.66, 8.87, 9.09, 9.31, 9.53, 9.76])
e48 = np.asarray([
       1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54, 1.62, 1.69,
       1.78, 1.87, 1.96, 2.05, 2.15, 2.26, 2.37, 2.49, 2.61, 2.74, 2.87, 3.01,
       3.16, 3.32, 3.48, 3.65, 3.83, 4.02, 4.22, 4.42, 4.64, 4.87, 5.11, 5.36,
       5.62, 5.90, 6.19, 6.49, 6.81, 7.15, 7.50, 7.87, 8.25, 8.66, 9.09, 9.53])
e24 = np.asarray([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                  3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])
e12 = np.asarray([1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2])

def current_through_resistor(resistor, voltage) -> Unit("A"):
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
    resistor = normalize_numeric(resistor)
    voltage = normalize_numeric(voltage)
    return voltage / resistor

def voltage_across_resistor(resistor, current) -> Unit("A"):
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
    resistor = normalize_numeric(resistor)
    current = normalize_numeric(current)
    return resistor * current

def power_dissipated_in_resistor_by_current(resistor, current) -> Unit("W"):
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
    resistor = normalize_numeric(resistor)
    current = normalize_numeric(current)
    return np.abs(resistor * current * current)

def power_dissipated_in_resistor_by_voltage(resistor, voltage) -> Unit("W"):
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
    resistor = normalize_numeric(resistor)
    voltage = normalize_numeric(voltage)
    current = current_through_resistor(resistor, voltage)
    return np.abs(current * voltage)

def resistor_range(multiplicator, sequence=e96):
    """
    Get a single range of resistors of a given sequence,
    e.g. for 1k to <10k use multiplicator = 1000.
    """
    return sequence * multiplicator

def standard_resistors(minExp=-1, maxExp=9, sequence=e96):
    """
    Get a list of all standard resistor values from 100mOhm up to 976 MΩ in Ω"""
    exponents = itertools.islice(itertools.count(minExp, 1), 0, maxExp - minExp)
    multiplicators = [10 ** x for x in exponents]
    return itertools.chain(*(resistor_range(r, sequence=sequence) for r in multiplicators))

def nearest_resistor(value, sequence=e96) -> Unit("Ω"):
    """
    Find the standard reistor value with the minimal difference to the given value
    """
    return min(standard_resistors(sequence=sequence), key=lambda r: abs(value - r))

def parallel_resistors(*args) -> Unit("Ω"):
    """
    Compute the total resistance of n parallel resistors and return
    the value in Ohms.
    """
    resistors = np.asarray(list(map(normalize_numeric, args)))
    # Check for zeros
    if len(np.nonzero(resistors)[0]) != len(resistors):
        return 0.0
    return 1.0 / np.sum(np.reciprocal(resistors.astype(float)))

def series_resistors(*args) -> Unit("Ω"):
    """
    Compute the total resistance of n parallel resistors and return
    the value in Ohms.
    """
    resistors = list(map(normalize_numeric, args))
    return sum(resistors)


def resistor_by_voltage_and_current(voltage, current) -> Unit("Ω"):
    """
    Compute the resistance value in Ohms that draws the given amount of
    current if the given voltage is across it.
    """
    voltage = normalize_numeric(voltage)
    current = normalize_numeric(current)
    return voltage / current
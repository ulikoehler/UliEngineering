#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
from UliEngineering.EngineerIO import normalize_numeric, normalize_timespan
import numpy as np
_ln2 = np.log(2)

__all__ = [
    "half_lifes_passed",
    "fraction_remaining",
    "fraction_decayed",
    "remaining_quantity",
    "decayed_quantity",
    "half_life_from_decay_constant",
    "half_life_from_remaining_quantity",
    "half_life_from_decayed_quantity",
    "half_life_from_fraction_remaining",
    "half_life_from_fraction_decayed",
]

def half_lifes_passed(timespan, half_life) -> float:
    """
    Compute the number of half-lifes that have passed within a certain
    timespan. The timespan can be a string or a number (in seconds).

    Examples:
        half_lifes_passed("1h", half_life="1min") => 1/60
    """
    timespan = normalize_timespan(timespan)
    half_life = normalize_timespan(half_life)
    return timespan / half_life

def fraction_remaining(timespan, half_life) -> float:
    """
    Compute the fraction of the original quantity that remains after
    a certain timespan and half-life.

    Examples:
        fraction_remaining("1h", half_life="1min") => 0.5
    """
    return 0.5 ** half_lifes_passed(timespan, half_life)

def fraction_decayed(timespan, half_life) -> float:
    """
    Compute the fraction of the original quantity that has decayed
    after a certain timespan and half-life.

    Examples:
        fraction_decayed("1h", half_life="1min") => 0.5
    """
    return 1.0 - fraction_remaining(timespan, half_life)

def remaining_quantity(timespan, half_life, initial_quantity) -> float:
    """
    Compute the quantity that remains after a certain timespan and half-life.

    Examples:
        remaining_quantity("1h", half_life="1min", initial_quantity=100) => ...
    """
    initial_quantity = normalize_numeric(initial_quantity)
    return fraction_remaining(timespan, half_life) * initial_quantity

def decayed_quantity(timespan, half_life, initial_quantity) -> float:
    """
    Compute the quantity that remains after a certain timespan and half-life.

    Examples:
        decayed_quantity("1h", half_life="1min", initial_quantity=100) => 40/60
    """
    initial_quantity = normalize_numeric(initial_quantity)
    return fraction_decayed(timespan, half_life) * initial_quantity

def half_life_from_decay_constant(decay_constant) -> float:
    """
    Compute the half-life from a decay constant using the formula: T₁/₂ = ln(2) / λ

    The half-life is the time required for a quantity to reduce to half of its initial value.
    This is commonly used in radioactive decay, drug elimination, and chemical kinetics.

    Parameters:
        decay_constant (float): The decay constant λ (lambda) in s⁻¹

    Returns:
        float: The half-life in the same time units as the decay constant (typically seconds)

    Example:
        >>> half_life_from_decay_constant(0.1)  # For λ = 0.1 s⁻¹
        6.9314718056
    """
    decay_constant = normalize_numeric(decay_constant)
    return _ln2 / decay_constant

def half_life_from_remaining_quantity(timespan, remaining_quantity, initial_quantity) -> float:
    """
    Compute the half-life from a remaining quantity after a certain timespan.

    Parameters:
        timespan (str or float): The timespan in seconds or a string like "1h"
        remaining_quantity (float): The quantity that remains after the timespan
        initial_quantity (float): The initial quantity

    Returns:
        float: The half-life in the same time units as the timespan

    Example:
        >>> half_life_from_remaining_quantity("1h", 50, 100)
        3600.0
    """
    timespan = normalize_timespan(timespan)
    remaining_quantity = normalize_numeric(remaining_quantity)
    initial_quantity = normalize_numeric(initial_quantity)
    return -timespan / (np.log(remaining_quantity / initial_quantity)/_ln2)

def half_life_from_decayed_quantity(timespan, decayed_quantity, initial_quantity) -> float:
    """
    Compute the half-life from a decayed quantity after a certain timespan.

    Parameters:
        timespan (str or float): The timespan in seconds or a string like "1h"
        decayed_quantity (float): The quantity that has decayed after the timespan
        initial_quantity (float): The initial quantity

    Returns:
        float: The half-life in the same time units as the timespan

    Example:
        >>> half_life_from_decayed_quantity("1h", 50, 100)
        3600.0
    """
    timespan = normalize_timespan(timespan)
    decayed_quantity = normalize_numeric(decayed_quantity)
    initial_quantity = normalize_numeric(initial_quantity)
    
    return -timespan / (np.log(1-decayed_quantity / initial_quantity)/_ln2)

def half_life_from_fraction_remaining(timespan, fraction_remaining) -> float:
    """
    Compute the half-life from a remaining fraction after a certain timespan.

    Parameters:
        timespan (str or float): The timespan in seconds or a string like "1h"
        fraction_remaining (float): The fraction of the initial quantity that remains

    Returns:
        float: The half-life in the same time units as the timespan

    Example:
        >>> half_life_from_fraction_remaining("1h", 0.5)
        3600.0
    """
    return half_life_from_remaining_quantity(timespan, fraction_remaining, 1.0)

def half_life_from_fraction_decayed(timespan, fraction_decayed) -> float:
    """
    Compute the half-life from a decayed fraction after a certain timespan.

    Parameters:
        timespan (str or float): The timespan in seconds or a string like "1h"
        fraction_decayed (float): The fraction of the initial quantity that has decayed

    Returns:
        float: The half-life in the same time units as the timespan

    Example:
        >>> half_life_from_fraction_decayed("1h", 0.5)
        3600.0
    """
    return half_life_from_decayed_quantity(timespan, fraction_decayed, 1.0)
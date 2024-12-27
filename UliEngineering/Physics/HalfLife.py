#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
from UliEngineering.EngineerIO import normalize_timespan

__all__ = ["half_lifes_passed", "fraction_remaining", "fraction_decayed", "remaining_quantity", "decayed_quantity"]

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
    return fraction_remaining(timespan, half_life) * initial_quantity

def decayed_quantity(timespan, half_life, initial_quantity) -> float:
    """
    Compute the quantity that remains after a certain timespan and half-life.

    Examples:
        decayed_quantity("1h", half_life="1min", initial_quantity=100) => 40/60
    """
    return fraction_decayed(timespan, half_life) * initial_quantity
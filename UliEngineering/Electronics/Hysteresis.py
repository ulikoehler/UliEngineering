#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding comparator / opamp hysteresis

For a detailed description please see http://www.ti.com/lit/ug/tidu020a/tidu020a.pdf
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Electronics.Resistors import parallel_resistors
from UliEngineering.Electronics.VoltageDivider import unloaded_ratio
import numpy as np

__all__ = ["hysteresis_threshold_ratios", "hysteresis_threshold_voltages",
           "hysteresis_threshold_ratios_opendrain",
           "hysteresis_threshold_voltages_opendrain"]


def hysteresis_threshold_ratios(r1, r2, rh):
    """
    Calculates hysteresis threshold factors for push-pull comparators

    Assumes that r1 and r2 are used to divide Vcc using a fixed ratio to
    obtain a threshold voltage.
    Additionally, Rh sources or sinks current to the threshold voltage,
    depending on the current state of the comparator.
    Additionally it it assumed that the same voltage (Vcc) that feeds
    the R1+R2 divider is output from the comparator and input to Rh.

    This function computes the (lower, upper) division ratios by
    assuming rh is set either parallel with R1 (upper) or with R2 (lower).

    Returns a tuple (lower, upper) containing floats
    representing the division ratios.

    Parameters
    ----------
    r1 : float or EngineerIO string
        The top resistor of the divider
    r2 : float or EngineerIO string
        The bottom resistor of the divider
    rh : float or EngineerIO string
        The hysteresis resistor of the divider
    """
    # Normalize inputs
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    rh = normalize_numeric(rh)
    # Compute r1, r2 in parallel with rh
    r1rh = parallel_resistors(r1, rh)
    r2rh = parallel_resistors(r2, rh)
    # Compute thresholds
    thl = unloaded_ratio(r1, r2rh)
    thu = unloaded_ratio(r1rh, r2)
    return (thl, thu)


def hysteresis_threshold_ratios_opendrain(r1, r2, rh):
    """
    Same as hysteresis_threshold_ratios(), but for open-drain comparators.
    In contrast to hysteresis_threshold_ratios(), ignores rh for the upper
    threshold.

    Parameters
    ----------
    r1 : float or EngineerIO string
        The top resistor of the divider
    r2 : float or EngineerIO string
        The bottom resistor of the divider
    rh : float or EngineerIO string
        The hysteresis resistor of the divider
    """
    # Normalize inputs
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    rh = normalize_numeric(rh)
    # Compute r1, r2 in parallel with rh
    r2rh = parallel_resistors(r2, rh)
    # Compute thresholds
    thl = unloaded_ratio(r1, r2rh)
    thu = unloaded_ratio(r1, r2)
    return (thl, thu)


def hysteresis_threshold_voltages(r1, r2, rh, vcc):
    """
    Same as hysteresis_threshold_ratios(), but calculates actual
    voltages instead of ratios.

    Returns (lower voltage, upper voltage), a tuple of floats.

    Parameters
    ----------
    r1 : float or EngineerIO string
        The top resistor of the divider
    r2 : float or EngineerIO string
        The bottom resistor of the divider
    rh : float or EngineerIO string
        The hysteresis resistor of the divider
    vcc : float or EngineerIO string
        The supply voltage that drives the output of the comparator
        and the R1/R2 network.
    """
    vcc = normalize_numeric(vcc)
    thl, thu = hysteresis_threshold_ratios(r1, r2, rh)
    return (thl * vcc, thu * vcc)


def hysteresis_threshold_voltages_opendrain(r1, r2, rh, vcc):
    """
    Same as hysteresis_threshold_ratios_opendrain(), but calculates actual
    voltages instead of ratios.

    Returns (lower voltage, upper voltage), a tuple of floats.

    Parameters
    ----------
    r1 : float or EngineerIO string
        The top resistor of the divider
    r2 : float or EngineerIO string
        The bottom resistor of the divider
    rh : float or EngineerIO string
        The hysteresis resistor of the divider
    vcc : float or EngineerIO string
        The supply voltage that drives the output of the comparator
        and the R1/R2 network.
    """
    vcc = normalize_numeric(vcc)
    thl, thu = hysteresis_threshold_ratios_opendrain(r1, r2, rh)
    return (thl * vcc, thu * vcc)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities regarding comparator / opamp hysteresis

For a detailed description please see http://www.ti.com/lit/ug/tidu020a/tidu020a.pdf
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Electronics.Resistors import parallel_resistors
from UliEngineering.Electronics.VoltageDivider import voltage_divider_ratio, bottom_resistor_by_ratio

__all__ = ["hysteresis_threshold_ratios", "hysteresis_threshold_voltages",
           "hysteresis_threshold_ratios_opendrain",
           "hysteresis_threshold_voltages_opendrain",
           "hysteresis_threshold_factors",
           "hysteresis_threshold_factors_opendrain",
           "hysteresis_resistor"]


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
    thl = voltage_divider_ratio(r1, r2rh)
    thu = voltage_divider_ratio(r1rh, r2)
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
    thl = voltage_divider_ratio(r1, r2rh)
    thu = voltage_divider_ratio(r1, r2)
    return (thl, thu)

def __hysteresis_threshold_voltages(r1, r2, rh, vcc, fn):
    """Internal push-pull & open-drain common code"""
    vcc = normalize_numeric(vcc)
    thl, thu = fn(r1, r2, rh)
    return (thl * vcc, thu * vcc)


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
    return __hysteresis_threshold_voltages(
        r1, r2, rh, vcc, hysteresis_threshold_ratios)

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
    return __hysteresis_threshold_voltages(
        r1, r2, rh, vcc, hysteresis_threshold_ratios_opendrain)


def __hysteresis_threshold_factors(r1, r2, rh, fn):
    """Internal push-pull & open-drain common code"""
    # Normalize inputs
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    rh = normalize_numeric(rh)
    # Compute thresholds
    thl, thu = fn(r1, r2, rh)
    # Compute factors
    thnom = voltage_divider_ratio(r1, r2)
    return (thl / thnom, thu / thnom)


def hysteresis_threshold_factors(r1, r2, rh):
    """
    Same as hysteresis_threshold_ratios(), but calculates the
    factor (nominal R1+R2 division ratio / actual ratio) for
    both the lower and the upper threshold instead of the ratios

    Returns (lower factor, upper factor), a tuple of floats.

    This is useful e.g. for computing

    Parameters
    ----------
    r1 : float or EngineerIO string
        The top resistor of the divider
    r2 : float or EngineerIO string
        The bottom resistor of the divider
    rh : float or EngineerIO string
        The hysteresis resistor of the divider
    """
    return __hysteresis_threshold_factors(
        r1, r2, rh, hysteresis_threshold_ratios)


def hysteresis_threshold_factors_opendrain(r1, r2, rh):
    """
    Same as hysteresis_threshold_ratios_opendrain(), but calculates the
    factor (nominal R1+R2 division ratio / actual ratio) for
    both the lower and the upper threshold instead of the ratios

    Returns (lower factor, upper factor), a tuple of floats.

    This is useful e.g. for computing

    Parameters
    ----------
    r1 : float or EngineerIO string
        The top resistor of the divider
    r2 : float or EngineerIO string
        The bottom resistor of the divider
    rh : float or EngineerIO string
        The hysteresis resistor of the divider
    """
    return __hysteresis_threshold_factors(
        r1, r2, rh, hysteresis_threshold_ratios_opendrain)


def hysteresis_resistor(r1, r2, fh=0.05):
    """
    Computes the hysteresis resistor Rh for a given
    R1, R2 divider network and a given deviation factor.

    The deviation factor fh represents the one-sided deviation
    from the nominal R1/R2 ratio. The total hysteresis is +-fh,
    i.e 2*fh.

    For example, for fh=0.05, the threshold will be 95% and 105%
    of the nominal ratio respectively.

    For open-drain comparators, fh represents the full deviation
    as the upper threshold is equivalent to the nominal threshold.


    Parameters
    ----------
    r1 : float or EngineerIO string
        The top resistor of the divider
    r2 : float or EngineerIO string
        The bottom resistor of the divider
    fh : float or EngineerIO string
        The deviation factor (e.g. 0.05 for 5% one-sided hysteresis
         deviation from the nominal r1/r2 value)
    """
    # Normalize inputs
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    fh = normalize_numeric(fh)
    # NOTE: We compute rh for the lower threshold only
    thnom = voltage_divider_ratio(r1, r2)
    ratio_target = thnom * (1. - fh)
    # Compute the resistor that, in parallel to R2, yields
    # a divider with our target ratio
    r2total = bottom_resistor_by_ratio(r1, ratio_target)
    # Solve 1/R3 = (1/R1 + 1/R2) for R2 => R2 = (R1 * R3) / (R1 - R3)
    return (r2 * r2total) / (r2 - r2total)

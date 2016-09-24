#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing different aspects and complexities of voltage dividers
"""
from UliEngineering.EngineerIO import normalize_numeric, Quantity
from .Resistors import *

__all__ = ["unloaded_ratio", "loaded_ratio", "top_resistor_by_ratio",
           "bottom_resistor_by_ratio", "feedback_top_resistor",
           "feedback_bottom_resistor", "feedback_actual_voltage"]


def unloaded_ratio(r1, r2) -> Quantity(""):
    """
    Compute the denominator of the  division ratio of a voltage divider, not taking into account
    parasitic properties or loading
    """
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    return r1 / (r1 + r2)

def loaded_ratio(r1, r2, rl) -> Quantity(""):
    """
    Compute the denominator of the  division ratio of a voltage divider, not taking into account
    parasitic properties but loading.
    """
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    rl = normalize_numeric(rl)
    return r1 / (r1 + parallel_resistors(r2, rl))


def top_resistor_by_ratio(rbottom, ratio) -> Quantity("Ω"):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rbottom = normalize_numeric(rbottom)
    ratio = normalize_numeric(ratio)
    return -(rbottom * ratio) / (ratio - 1.0)


def bottom_resistor_by_ratio(rtop, ratio) -> Quantity("Ω"):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rtop = normalize_numeric(rtop)
    ratio = normalize_numeric(ratio)
    return rtop * (1.0 / ratio - 1.0)

def feedback_top_resistor(vexp, rbot, vfb) -> Quantity("Ω"):
    """
    Utility to compute the top feedback resistor
    in a voltage feedback network (e.g. for a DC/DC converter)

    Parameters
    ----------
    vexp : float
        The voltage at between top and bottom of the voltage divider
    rbot : float
        The known bottom resistor
    vfb : float
        The feedback voltage that is servoed by the regulator
    """
    vexp = normalize_numeric(vexp)
    rbot = normalize_numeric(rbot)
    vfb = normalize_numeric(vfb)
    # Vo = Vfb * (R1/R2 + 1)
    # solve A = B*((C/D) + 1) for C
    return rbot * (vexp - vfb) / vfb

def feedback_bottom_resistor(vexp, rtop, vfb) -> Quantity("Ω"):
    """
    Utility to compute the bottom feedback resistor
    in a voltage feedback network (e.g. for a DC/DC converter)

    Parameters
    ----------
    vexp : float
        The voltage at between top and bottom of the voltage divider
    rtop : float
        The known top resistor
    vfb : float
        The feedback voltage that is servoed by the regulator
    """
    vexp = normalize_numeric(vexp)
    rtop = normalize_numeric(rtop)
    vfb = normalize_numeric(vfb)
    # Vo = Vfb * (R1/R2 + 1)
    # solve A = B*((C/D) + 1) for D
    return (vfb * rtop) / (vexp - vfb)


def feedback_actual_voltage(rtop, rbot, vfb) -> Quantity("V"):
    """
    Compute the actual voltage regulator output in a feedback
    servo setup. Returns the Vout voltage.

    Parameters
    ----------
    rtop : float
        The top resistor of the voltage divider
    rbot : float
        The bottom resistor of the voltage divider
    vfb : float
        The feedback voltage
    """
    # Equation: Vout * ratio = vfb
    ratio = unloaded_ratio(rtop, rbot)
    return normalize_numeric(vfb) / ratio



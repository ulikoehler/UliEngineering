#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing different aspects and complexities of voltage dividers
"""
from UliEngineering.EngineerIO import normalize_numeric
from .Resistors import *
import numpy as np
from UliEngineering.Units import Unit

__all__ = ["voltage_divider_ratio", "top_resistor_by_ratio",
           "bottom_resistor_by_ratio", "feedback_top_resistor",
           "feedback_bottom_resistor", "feedback_actual_voltage"]


def voltage_divider_ratio(r1, r2, rload=np.inf) -> Unit(""):
    """
    Compute the denominator of the  division ratio of a voltage divider.
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    return r2 / (r1 + parallel_resistors(r2, rload))

def top_resistor_by_ratio(rbottom, ratio) -> Unit("Ω"):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rbottom = normalize_numeric(rbottom)
    ratio = normalize_numeric(ratio)
    return rbottom * (1.0 / ratio - 1.0)

def bottom_resistor_by_ratio(rtop, ratio) -> Unit("Ω"):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rtop = normalize_numeric(rtop)
    ratio = normalize_numeric(ratio)
    return -(rtop * ratio) / (ratio - 1.0)

def feedback_top_resistor(vexp, rbot, vfb) -> Unit("Ω"):
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

def feedback_bottom_resistor(vexp, rtop, vfb) -> Unit("Ω"):
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

def feedback_actual_voltage(rtop, rbot, vfb) -> Unit("V"):
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
    ratio = voltage_divider_ratio(rtop, rbot)
    return normalize_numeric(vfb) / ratio

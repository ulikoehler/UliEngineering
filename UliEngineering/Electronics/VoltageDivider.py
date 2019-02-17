#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing different aspects and complexities of voltage dividers
"""
from UliEngineering.EngineerIO import normalize_numeric, format_value
from .Resistors import *
import numpy as np
from collections import namedtuple
from UliEngineering.Units import Unit

__all__ = ["voltage_divider_ratio", "top_resistor_by_ratio",
           "voltage_divider_voltage", "voltage_divider_current",
           "bottom_resistor_by_ratio", "feedback_top_resistor",
           "feedback_bottom_resistor", "feedback_actual_voltage",
           "voltage_divider_power"]


def voltage_divider_ratio(rtop, rbot, rload=np.inf) -> Unit(""):
    """
    Compute the division ratio of a voltage divider.
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
    rtop = normalize_numeric(rtop)
    rbot = normalize_numeric(rbot)
    return rbot / (rtop + parallel_resistors(rbot, rload))

def voltage_divider_voltage(rtop, rbot, vin, rload=np.inf) -> Unit("V"):
    """
    Compute the voltage output of a voltage divider
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
    vin = normalize_numeric(vin)
    return voltage_divider_ratio(rtop, rbot, rload=rload) * vin

def voltage_divider_current(rtop, rbot, vin, rload=np.inf) -> Unit("V"):
    """
    Compute the current through the top resistor of a voltage divider.
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
    vin = normalize_numeric(vin)
    rtop = normalize_numeric(rtop)
    vout = voltage_divider_voltage(rtop, rbot, vin, rload=rload)
    # Compute voltage delta across resistor
    vdelta = vout - vin
    # Compute current through resisotr
    return current_through_resistor(rtop, vdelta);

class VoltageDividerPower(namedtuple("VoltageDividerPower", [
        "top", "bottom", "load", "total"
    ])):
    """
    Represents the power dissipated in different parts of a voltage
    """
    def __repr__(self):
        """Better formatting"""
        return "VoltageDividerPower(top={}, bottom={}, {}total={})".format(
            format_value(self.top, "W"),
            format_value(self.bottom, "W"),
            format_value(self.load, "W") if self.load != 0 else "",
            format_value(self.total, "W")
        )

def voltage_divider_power(rtop, rbot, vin, rload=np.inf):
    """
    Compute the power dissipated in a voltage divider.

    Returns a VoltageDividerPower object.

    Usage example:
    >>> voltage_divider_power("250k", "1k", "230V")
    VoltageDividerPower(top=210 mW, bottom=840 µW, total=211 mW
    >>> voltage_divider_power("250k", "1k", "230V").total
    0.2107569721115538

    
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
    vin = normalize_numeric(vin)
    rtop = normalize_numeric(rtop)
    vout = voltage_divider_voltage(rtop, rbot, vin, rload=rload)
    # Compute voltage delta across resistor
    ptop = power_dissipated_in_resistor_by_voltage(rtop, vout - vin)
    pbot = power_dissipated_in_resistor_by_voltage(rbot, vout)
    pload = power_dissipated_in_resistor_by_voltage(rload, vout)
    return VoltageDividerPower(
        ptop, pbot, pload, ptop + pbot + pload
    )

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing different aspects and complexities of voltage dividers
"""
from UliEngineering.Electronics.Resistors import current_through_resistor, parallel_resistors, power_dissipated_in_resistor_by_voltage
from UliEngineering.EngineerIO import normalize_numeric, format_value
import numpy as np
from collections import namedtuple

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = ["voltage_divider_ratio", "top_resistor_by_ratio",
           "voltage_divider_voltage", "voltage_divider_current",
           "bottom_resistor_by_ratio", "feedback_top_resistor",
           "feedback_bottom_resistor", "feedback_actual_voltage",
           "voltage_divider_power"]

@returns_unit("")
@normalize_numeric_args
def voltage_divider_ratio(rtop, rbot, rload=np.inf):
    """
    Compute the division ratio of a voltage divider.
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
    return rbot / (rtop + parallel_resistors(rbot, rload))

@returns_unit("V")
@normalize_numeric_args
def voltage_divider_voltage(rtop, rbot, vin, rload=np.inf):
    """
    Compute the voltage output of a voltage divider
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
    return voltage_divider_ratio(rtop, rbot, rload=rload) * vin

@returns_unit("A")
@normalize_numeric_args
def voltage_divider_current(rtop, rbot, vin, rload=np.inf):
    """
    Compute the current through the top resistor of a voltage divider.
    
    If rload is supplied, additional load (in parallel to R2) is taken into account.
    """
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
        return f"VoltageDividerPower(top={format_value(self.top, 'W')}, bottom={format_value(self.bottom, 'W')}, {'load=' + format_value(self.load, 'W') if self.load != 0 else ''}total={format_value(self.total, 'W')})"

@returns_unit("W")
@normalize_numeric_args
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
    vout = voltage_divider_voltage(rtop, rbot, vin, rload=rload)
    # Compute voltage delta across resistor
    ptop = power_dissipated_in_resistor_by_voltage(rtop, vout - vin)
    pbot = power_dissipated_in_resistor_by_voltage(rbot, vout)
    pload = power_dissipated_in_resistor_by_voltage(rload, vout)
    return VoltageDividerPower(
        ptop, pbot, pload, ptop + pbot + pload
    )

@returns_unit("Ω")
@normalize_numeric_args
def top_resistor_by_ratio(rbottom, ratio):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rbottom = normalize_numeric(rbottom)
    ratio = normalize_numeric(ratio)
    return rbottom * (1.0 / ratio - 1.0)

@returns_unit("Ω")
@normalize_numeric_args
def bottom_resistor_by_ratio(rtop, ratio):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rtop = normalize_numeric(rtop)
    ratio = normalize_numeric(ratio)
    return -(rtop * ratio) / (ratio - 1.0)

@returns_unit("Ω")
@normalize_numeric_args
def feedback_top_resistor(vexp, rbot, vfb, rload=np.inf):
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
    rload : float
        A load resistor in parallel to the bottom resistor
    """
    # Vo = Vfb * (R1/R2 + 1)
    # solve A = B*((C/D) + 1) for C
    return (parallel_resistors(rbot, rload)) * (vexp - vfb) / vfb

@returns_unit("Ω")
@normalize_numeric_args
def feedback_bottom_resistor(vexp, rtop, vfb):
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
    # Vo = Vfb * (R1/R2 + 1)
    # solve A = B*((C/D) + 1) for D
    return (vfb * rtop) / (vexp - vfb)

@returns_unit("V")
@normalize_numeric_args
def feedback_actual_voltage(rtop, rbot, vfb, rload=np.inf):
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
    ratio = voltage_divider_ratio(rtop, parallel_resistors(rbot, rload))
    return vfb / ratio

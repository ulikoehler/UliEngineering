#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for LED calculations

Usage example:
>>> from UliEngineering.Electronics.OpAmp import summing_amplifier_noninv
>>> # Example: sum 2.5V and 0.5V with a total sum-referred gain of 1.0
>>> formatValue(summing_amplifier_noninv(
        "2.5V", "500mV", "1kΩ", "1kΩ", "1kΩ", "1kΩ"), "V"))

"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Exceptions import OperationImpossibleException
from UliEngineering.Units import Unit
from UliEngineering.Electronics.Resistors import resistor_current_by_power

__all__ = [
    "LEDForwardVoltages",
    "led_series_resistor",
    "led_series_resistor_power",
    "led_series_resistor_maximum_current",
    "led_series_resistor_current",
]


class LEDForwardVoltages():
    """
    Common LED forward voltage values.
    Source: http://www.elektronik-kompendium.de/sites/bau/1109111.htm
    NOTE: These do NOT neccessarily represent the actual forward voltages
    of any LED you choose but rather the typical forward voltage at nominal
    current.

    Note that diode testers test the forward voltage with rather low currents
    and the forward voltage might vary slightly at operating current.
    Take that into account when operating a LED near its maximum allowed current.
    """
    Infrared = 1.5
    Red = 1.6
    Yellow = 2.2
    Green = 2.1
    Blue = 2.9
    White = 4.0

def led_series_resistor(vsupply, ioperating, vforward) -> Unit("Ω"):
    """
    Computes the required series resistor for operating a LED with
    forward voltage [vforward] at current [ioperating] on a
    supply voltage of [vsupply].

    Tolerances are not taken into account.
    """
    vsupply = normalize_numeric(vsupply)
    ioperating = normalize_numeric(ioperating)
    vforward = normalize_numeric(vforward)
    if vforward > vsupply:
        raise OperationImpossibleException(
            f"Can't operate LED with forward voltage {vforward} on {vsupply} supply"
        )
    return (vsupply - vforward) / ioperating

def led_series_resistor_power(vsupply, ioperating, vforward) -> Unit("W"):
    """
    Computes the required series resistor power for operating a LED with
    forward voltage [vforward] at current [ioperating] on a
    supply voltage of [vsupply].
    
    The resulting power value is the minimum rated value for the resistor
    for continous operation
 
    Tolerances are not taken into account.
    """
    vsupply = normalize_numeric(vsupply)
    ioperating = normalize_numeric(ioperating)
    vforward = normalize_numeric(vforward)
    if vforward > vsupply:
        raise OperationImpossibleException(
            f"Can't operate LED with forward voltage {vforward} on {vsupply} supply"
        )
    # Will raise OperationImpossibleException if vforward > vsupply
    resistor_value = led_series_resistor(vsupply, ioperating, vforward)
    return resistor_value * ioperating * ioperating

def led_series_resistor_maximum_current(resistance, power_rating) -> Unit("A"):
    """
    Compute the maximum current through a LED + series resistor combination,
    so that the power rating of the resistor is not exceeded
    (i.e. the current where the dissipated power is exactly the power rating).

    Tolerances are not taken into account.
    """
    power_rating = normalize_numeric(power_rating)
    resistance = normalize_numeric(resistance)
    # Compute the current that would flow through the resistor
    current = resistor_current_by_power(resistance, power_rating)
    return current

def led_series_resistor_current(vsupply, resistance, vforward) -> Unit("A"):
    """
    Compute the current that flows through a LED + series resistor combination
    when connected to a supply voltage [vsupply] and a series resistor of [resistance].

    Tolerances are not taken into account.
    """
    vsupply = normalize_numeric(vsupply)
    resistance = normalize_numeric(resistance)
    vforward = normalize_numeric(vforward)
    if vforward > vsupply:
        raise OperationImpossibleException(
            f"Can't operate LED with forward voltage {vforward} on {vsupply} supply"
        )
    return (vsupply - vforward) / resistance

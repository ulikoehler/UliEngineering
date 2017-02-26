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

__all__ = ["LEDForwardVoltages", "led_series_resistor"]


class LEDForwardVoltages():
    """
    Common LED forward voltage values.
    Source: http://www.elektronik-kompendium.de/sites/bau/1109111.htm
    NOTE: These do NOT neccessarily represent the actual forward voltages
    of any LED you choose but rather the typical forward voltage at nominal
    current.

    Note that diode testers test the forward voltage with rather low currents
    and the forward voltage might vary slightly at operating current.
    Take that into account when operating a LED near its maximum values.
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
            "Can't operate LED with forward voltage {} on {} supply".format(
                vsupply, vforward
            ))
    return (vsupply - vforward) / ioperating
